using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public class GeneratorSession
{
    const string GeneratorScene = "Assets/Scenes/Generator.unity";

    static GeneratorSession _active;

    readonly bool _quitWhenDone;
    readonly bool _verbose;

    int    _exitCode;
    double _deadline;
    string _outDir;

    SimulationDirector  _director;
    MultiCameraRecorder _recorder;

    // Saved so we can restore the project's play-mode settings on exit.
    bool                 _prevOptionsEnabled;
    EnterPlayModeOptions _prevOptions;

    GeneratorSession(bool quitWhenDone, bool verbose)
    {
        _quitWhenDone = quitWhenDone;
        _verbose      = verbose;
    }

    public static void Begin(string config, string outDir, bool quitWhenDone, bool verbose)
    {
        if (_active != null)
        {
            Debug.LogError("[Sim] A generation run is already in progress.");
            if (quitWhenDone) EditorApplication.Exit(1);
            return;
        }

        _active = new GeneratorSession(quitWhenDone, verbose);
        _active.Launch(config, outDir);
    }

    // Lifecycle
    void Launch(string config, string outDir)
    {
        _exitCode = 0;
        try
        {
            if (!File.Exists(GeneratorScene))
            {
                Debug.LogError($"[Sim] {GeneratorScene} not found. Build it first: " +
                               "menu Tools > MaritimeSim > Build Generator Scene.");
                _exitCode = 3;
                Conclude();
                return;
            }

            SessionState.SetString("MaritimeSim.config", Path.GetFullPath(config));
            SessionState.SetString("MaritimeSim.out",
                string.IsNullOrEmpty(outDir) ? "" : Path.GetFullPath(outDir));

            _prevOptionsEnabled = EditorSettings.enterPlayModeOptionsEnabled;
            _prevOptions        = EditorSettings.enterPlayModeOptions;
            EditorSettings.enterPlayModeOptionsEnabled = true;
            EditorSettings.enterPlayModeOptions =
                EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;

            EditorSceneManager.OpenScene(GeneratorScene, OpenSceneMode.Single);

            EditorApplication.playModeStateChanged += OnPlayModeChanged;
            EditorApplication.EnterPlaymode();
        }
        catch (Exception e)
        {
            Debug.LogError($"[Sim] Launch failed: {e}");
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            RestorePlayModeOptions();
            _exitCode = 1;
            Conclude();
        }
    }

    void OnPlayModeChanged(PlayModeStateChange state)
    {
        try
        {
            switch (state)
            {
                case PlayModeStateChange.EnteredPlayMode:
                    StartRecording();
                    break;

                case PlayModeStateChange.EnteredEditMode:
                    Finish();
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[Sim] Play-mode handler failed: {e}");
            _exitCode = 1;
            if (EditorApplication.isPlaying)
            {
                EditorApplication.update -= OnUpdate;
                EditorApplication.isPlaying = false;
            }
            else
            {
                Finish();
            }
        }
    }

    void StartRecording()
    {
        _director = UnityEngine.Object.FindObjectOfType<SimulationDirector>();
        if (_director == null)
            throw new Exception("No SimulationDirector found in the Generator scene.");
        if (_director.BuildFailed)
            throw new Exception("SimulationDirector failed to build the scene (see earlier errors).");
        if (_director.Config == null || _director.Cameras.Count == 0)
            throw new Exception("SimulationDirector produced no cameras (see earlier errors).");

        _recorder = new MultiCameraRecorder(_director, _verbose);
        _recorder.Start();

        _director.BeginCapture();

        // Wall-clock watchdog so a stalled run can never hang the editor forever.
        double seconds = _director.TotalFrames / (double)Mathf.Max(1, _director.Config.video.fps);
        _deadline = EditorApplication.timeSinceStartup + seconds * 20.0 + 120.0;

        EditorApplication.update += OnUpdate;
    }

    void OnUpdate()
    {
        bool finished = _director == null || _director.IsFinished;
        bool timedOut = EditorApplication.timeSinceStartup > _deadline;
        if (!finished && !timedOut) return;

        EditorApplication.update -= OnUpdate;

        if (timedOut && !finished)
        {
            Debug.LogError("[Sim] Timed out waiting for capture to finish.");
            _exitCode = 1;
        }

        try
        {
            if (_director != null) _outDir = _director.OutDir;   // remember for the post-play log

            // Finalize while STILL in play mode: Stop() flushes/encodes each mp4
            // synchronously, and the director writes its ground-truth files.
            _recorder?.Stop();
            _director?.FinalizeGroundTruth();
            Debug.Log("[Sim] Recording complete. Videos + ground truth written.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[Sim] Error while finalizing: {e}");
            _exitCode = 1;
        }

        EditorApplication.isPlaying = false;
    }

    void Finish()
    {
        EditorApplication.playModeStateChanged -= OnPlayModeChanged;
        RestorePlayModeOptions();
        Conclude();
    }

    void Conclude()
    {
        _active = null;

        if (_quitWhenDone)
        {
            EditorApplication.Exit(_exitCode);
            return;
        }

        if (_exitCode == 0)
            Debug.Log($"[Sim] Done — videos + ground truth written to:\n{_outDir}");
        else
            Debug.LogError("[Sim] Generation failed — see the console above for the cause.");
    }

    void RestorePlayModeOptions()
    {
        EditorSettings.enterPlayModeOptionsEnabled = _prevOptionsEnabled;
        EditorSettings.enterPlayModeOptions        = _prevOptions;
    }
}
