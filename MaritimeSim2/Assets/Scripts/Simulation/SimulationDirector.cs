using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

//  SimulationDirector.cs
//  Lives in Assets/Scenes/Generator.unity (placed by GeneratorSceneBuilder).
//  At play start it:
//      1. resolves the config path + output dir
//      2. wipes any stray vessels / cameras left in the scene
//      3. spawns the ships and cameras described by the config
//      4. gives one RenderTexture + one GroundTruthRecorder to each camera
//      5. exposes the cameras so MultiCameraRecorder can attach a movie recorder
//      6. drives the per-frame ground-truth capture, frame-aligned to the video
//
//  Determinism: BeginCapture() pins Time.captureDeltaTime to 1/fps so every
//  rendered frame advances the simulation by exactly one video frame, making
//  ship motion, wave height and ground truth reproducible run-to-run.

[DefaultExecutionOrder(-100)]   // build the scene before ShipControllers tick
public class SimulationDirector : MonoBehaviour
{
    [Header("Prefab registry (populated by GeneratorSceneBuilder)")]
    public GameObject[] shipPrefabs;

    [Header("Editor overrides")]
    [Tooltip("Config path for pressing Play in the Editor. CLI -simConfig wins over this.")]
    public string configPathOverride = "";
    [Tooltip("Output dir for pressing Play in the Editor. CLI -simOut wins over this.")]
    public string outDirOverride = "";

    public const string VesselTag = "Vessel";

    public class CameraRig
    {
        public string               name;
        public Camera               camera;
        public RenderTexture        renderTexture;
        public GroundTruthRecorder  groundTruth;
    }

    readonly List<CameraRig> _rigs = new List<CameraRig>();

    public IReadOnlyList<CameraRig> Cameras => _rigs;
    public SimConfig Config   { get; private set; }
    public string    OutDir   { get; private set; }
    public int       TotalFrames => Config != null ? Config.TotalFrames : 0;
    public bool      IsFinished  { get; private set; }
    public bool      BuildFailed { get; private set; }

    int  _frame;
    bool _capturing;

    void Awake()
    {
        try
        {
            string configPath = ResolveConfigPath();
            Config = SimConfig.Load(configPath);
            OutDir = ResolveOutDir(configPath);
            Directory.CreateDirectory(OutDir);

            BuildScene();
            SetupEnvironmentLighting();

            // copy the exact config next to the outputs.
            try { File.Copy(configPath, Path.Combine(OutDir, "config.json"), true); }
            catch (Exception e) { Debug.LogWarning($"[Sim] Could not copy config for provenance: {e.Message}"); }

            Debug.Log($"[Sim] Built scene: {Config.ships.Length} ship(s), {_rigs.Count} camera(s), " +
                      $"{TotalFrames} frames @ {Config.video.fps}fps -> {OutDir}");
        }
        catch (Exception e)
        {
            // Flag the failure and let GeneratorSession stop play mode cleanly.
            Debug.LogError($"[Sim] Failed to build scene: {e}");
            BuildFailed = true;
#if !UNITY_EDITOR
            Application.Quit(1);
#endif
        }
    }

    void BuildScene()
    {
        //  Grab a camera to clone,
        //  then clear the slate.
        Camera template = FindTemplateCamera();
        if (template == null)
            throw new Exception("Generator scene has no camera to use as a template. Rebuild it with Tools/MaritimeSim/Build Generator Scene.");
        template.gameObject.SetActive(false);

        ClearStrays(template.gameObject);

        SpawnShips();
        SpawnCameras(template);

        Destroy(template.gameObject);
    }

    Camera FindTemplateCamera()
    {
        Camera chosen = null;
        foreach (var cam in FindObjectsOfType<Camera>(true))
        {
            // Skip cameras owned by other systems (eg Crest's internal cameras)
            if (cam.hideFlags != HideFlags.None) continue;
            chosen = cam;
            break;
        }
        return chosen;
    }

    void ClearStrays(GameObject keep)
    {
        foreach (var go in GameObject.FindGameObjectsWithTag(VesselTag))
            Destroy(go);

        foreach (var cam in FindObjectsOfType<Camera>(true))
        {
            if (cam.gameObject == keep) continue;
            if (cam.hideFlags != HideFlags.None) continue;   // leave engine/Crest cameras alone
            Destroy(cam.gameObject);
        }
    }

    void SpawnShips()
    {
        var registry = BuildPrefabRegistry();

        foreach (var s in Config.ships)
        {
            if (!registry.TryGetValue(s.prefab, out var prefab))
                throw new Exception($"Ship prefab '{s.prefab}' not found. Available: {string.Join(", ", registry.Keys)}");

            var ship = Instantiate(prefab);
            ship.name = s.name;
            SetTagRecursivelyOff(ship);             // ensure only the root is the tracked vessel
            ship.tag = VesselTag;

            EnsureTightBounds(ship);

            var controller = ship.GetComponent<ShipController>();
            if (controller == null) controller = ship.AddComponent<ShipController>();
            controller.Init(s.Position(), s.heading, s.speed, s.classId);
        }
    }

    void SpawnCameras(Camera template)
    {
        for (int i = 0; i < Config.cameras.Length; i++)
        {
            var c = Config.cameras[i];

            var go = Instantiate(template.gameObject);
            go.name = $"cam_{c.name}";
            go.SetActive(true);
            go.tag = (i == 0) ? "MainCamera" : "Untagged";  // give Crest a single viewpoint

            go.transform.SetPositionAndRotation(c.Position(), c.Rotation());

            var cam = go.GetComponent<Camera>();
            cam.fieldOfView = c.fov;

            // Strip components we don't want duplicated across clones.
            var existingGt = go.GetComponent<GroundTruthRecorder>();
            if (existingGt != null) Destroy(existingGt);
            var listener = go.GetComponent<AudioListener>();
            if (listener != null) Destroy(listener);   // no audio capture; avoids multi-listener warnings

            // Dedicated off-screen target so every camera records independently.
            var rt = new RenderTexture(Config.video.width, Config.video.height, 24, RenderTextureFormat.Default)
            {
                name = $"RT_{c.name}",
                antiAliasing = 1,
            };
            rt.Create();
            cam.targetTexture = rt;

            var gt = go.AddComponent<GroundTruthRecorder>();
            gt.renderCamera  = cam;
            gt.outputFolder  = OutDir;
            gt.sequenceName  = c.name;
            gt.autoCapture   = false;          // the director drives CaptureFrame()
            gt.writeMOT      = true;
            gt.writeJSON     = true;

            _rigs.Add(new CameraRig { name = c.name, camera = cam, renderTexture = rt, groundTruth = gt });
        }
    }

    // Headless -executeMethod runs don't auto-generate the skybox-sourced ambient and reflection,
    // so the water renders dark and flat. Generate both explicitly.
    void SetupEnvironmentLighting()
    {
        DynamicGI.UpdateEnvironment();   // ambient from skybox

        // Specular reflection: render a scene-spanning skybox probe once.
        var go = new GameObject("__SkyReflectionProbe");
        var probe = go.AddComponent<ReflectionProbe>();
        probe.mode            = UnityEngine.Rendering.ReflectionProbeMode.Realtime;
        probe.refreshMode     = UnityEngine.Rendering.ReflectionProbeRefreshMode.ViaScripting;
        probe.timeSlicingMode = UnityEngine.Rendering.ReflectionProbeTimeSlicingMode.NoTimeSlicing;
        probe.clearFlags      = UnityEngine.Rendering.ReflectionProbeClearFlags.Skybox;
        probe.cullingMask     = 0;                         // sky only
        probe.size            = Vector3.one * 100000f;     // cover the scene
        probe.boxProjection   = false;
        probe.transform.position = Vector3.zero;
        probe.RenderProbe();
    }

    Dictionary<string, GameObject> BuildPrefabRegistry()
    {
        var map = new Dictionary<string, GameObject>(StringComparer.OrdinalIgnoreCase);
        if (shipPrefabs != null)
            foreach (var p in shipPrefabs)
                if (p != null) map[p.name] = p;
        return map;
    }


    //  Capture loop
    public void BeginCapture()
    {
        // Fixed-step time so motion + waves + ground truth are reproducible and
        // frame-aligned with the recorder, regardless of real render speed.
        Time.captureDeltaTime = 1f / Mathf.Max(1, Config.video.fps);
        _frame = 0;
        _capturing = true;
        IsFinished = false;
    }

    void LateUpdate()
    {
        if (!_capturing || IsFinished) return;

        _frame++;
        foreach (var rig in _rigs)
            rig.groundTruth.CaptureFrame(_frame);

        if (_frame >= TotalFrames)
        {
            _capturing = false;
            IsFinished = true;
        }
    }

    // Flush every ground-truth file. Call before exiting play mode
    public void FinalizeGroundTruth()
    {
        foreach (var rig in _rigs)
            rig.groundTruth.FinalizeAndClose();
        Time.captureDeltaTime = 0f;
    }

    //  Helpers
    string ResolveConfigPath()
    {
        string cli = GetArg("-simConfig");
        if (!string.IsNullOrEmpty(cli)) return cli;

#if UNITY_EDITOR
        string sess = UnityEditor.SessionState.GetString("MaritimeSim.config", "");
        if (!string.IsNullOrEmpty(sess)) return sess;
#endif
        if (!string.IsNullOrEmpty(configPathOverride)) return configPathOverride;

        throw new Exception("No config provided. Pass -simConfig <path> on the command line, " +
                            "or set 'Config Path Override' on the SimulationDirector for in-Editor testing.");
    }

    string ResolveOutDir(string configPath)
    {
        string outDir = GetArg("-simOut");
#if UNITY_EDITOR
        if (string.IsNullOrEmpty(outDir))
            outDir = UnityEditor.SessionState.GetString("MaritimeSim.out", "");
#endif
        if (string.IsNullOrEmpty(outDir))
            outDir = outDirOverride;
        if (string.IsNullOrEmpty(outDir))
            outDir = Path.Combine("Recordings", Path.GetFileNameWithoutExtension(configPath));

        if (!Path.IsPathRooted(outDir))
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            outDir = Path.GetFullPath(Path.Combine(projectRoot, outDir));
        }
        return outDir;
    }

    static string GetArg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.Ordinal))
                return args[i + 1];
        return null;
    }

    static void EnsureTightBounds(GameObject ship)
    {
        if (ship.GetComponent<TightBounds>() != null) return;

        var renderers = ship.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return;

        Bounds b = renderers[0].bounds;
        foreach (var r in renderers) b.Encapsulate(r.bounds);

        var tb = ship.AddComponent<TightBounds>();
        // Convert world AABB to the local box TightBounds expects.
        tb.center = ship.transform.InverseTransformPoint(b.center);
        Vector3 ls = ship.transform.lossyScale;
        tb.size = new Vector3(
            b.size.x / Mathf.Max(1e-4f, Mathf.Abs(ls.x)),
            b.size.y / Mathf.Max(1e-4f, Mathf.Abs(ls.y)),
            b.size.z / Mathf.Max(1e-4f, Mathf.Abs(ls.z)));
    }

    // Clear the Vessel tag from children so FindGameObjectsWithTag only returns roots.
    static void SetTagRecursivelyOff(GameObject root)
    {
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
            if (t.CompareTag(VesselTag))
                t.gameObject.tag = "Untagged";
    }
}
