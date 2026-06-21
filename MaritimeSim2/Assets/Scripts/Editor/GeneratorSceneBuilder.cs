using System.Collections.Generic;
using Crest;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class GeneratorSceneBuilder
{
    const string SourceScene  = "Assets/Scenes/Sample.unity";
    const string OutputScene  = "Assets/Scenes/Generator.unity";
    const string ShipPrefabDir = "Assets/ShipsPrefabs";

    [MenuItem("Tools/MaritimeSim/Build Generator Scene")]
    public static void Build()
    {
        var scene = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);

        RemoveShips(scene);
        Camera template = KeepSingleTemplateCamera(scene);
        if (template != null)
        {
            var gt = template.GetComponent<GroundTruthRecorder>();
            if (gt != null) Object.DestroyImmediate(gt);
            template.targetTexture = null;
        }
        else
        {
            Debug.LogError("[GenBuild] Sample.unity has no camera to use as a template.");
        }

        AddDirector();

        EditorSceneManager.MarkSceneDirty(scene);
        bool ok = EditorSceneManager.SaveScene(scene, OutputScene);
        AssetDatabase.Refresh();

        if (ok) Debug.Log($"[GenBuild] Wrote {OutputScene}");
        else    Debug.LogError($"[GenBuild] Failed to save {OutputScene}");
    }

    static void RemoveShips(Scene scene)
    {
        var toDelete = new HashSet<GameObject>();

        // Anything tagged as a vessel, or carrying a boat-ish component, is a ship.
        foreach (var go in scene.GetRootGameObjects())
        {
            if (go.CompareTag(SimulationDirector.VesselTag))
            {
                toDelete.Add(go);
                continue;
            }
            if (go.GetComponentInChildren<BoatProbes>(true) != null ||
                go.GetComponentInChildren<AutoForwardEngine>(true) != null ||
                go.GetComponentInChildren<ShipController>(true) != null)
            {
                toDelete.Add(go);
            }
        }

        foreach (var go in toDelete)
            Object.DestroyImmediate(go);
    }

    static Camera KeepSingleTemplateCamera(Scene scene)
    {
        var cameras = new List<Camera>();
        foreach (var go in scene.GetRootGameObjects())
            cameras.AddRange(go.GetComponentsInChildren<Camera>(true));

        if (cameras.Count == 0) return null;

        // Prefer the MainCamera-tagged one as the template.
        Camera template = cameras.Find(c => c.CompareTag("MainCamera")) ?? cameras[0];

        foreach (var cam in cameras)
            if (cam != template)
                Object.DestroyImmediate(cam.gameObject);

        return template;
    }

    static void AddDirector()
    {
        var go = new GameObject("__SimulationDirector");
        var director = go.AddComponent<SimulationDirector>();
        director.shipPrefabs = LoadShipPrefabs();
        EditorUtility.SetDirty(go);
    }

    static GameObject[] LoadShipPrefabs()
    {
        var guids = AssetDatabase.FindAssets("t:Prefab", new[] { ShipPrefabDir });
        var prefabs = new List<GameObject>();
        foreach (var guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab != null) prefabs.Add(prefab);
        }
        Debug.Log($"[GenBuild] Registered {prefabs.Count} ship prefab(s) from {ShipPrefabDir}.");
        return prefabs.ToArray();
    }
}
