using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class GeneratorEntry
{
    [MenuItem("Tools/MaritimeSim/Generate Videos from Config…")]
    public static void GenerateInteractive()
    {
        string rootDir = Directory.GetParent(Application.dataPath).FullName;
        string config = EditorUtility.OpenFilePanel("Select simulation config (JSON)", rootDir, "json");
        if (string.IsNullOrEmpty(config))
            return;
            
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            return;

        GeneratorSession.Begin(config, outDir: null, quitWhenDone: false, verbose: false);
    }

    // Command-line entry: Unity.exe -executeMethod GeneratorEntry.Run
    //   -simConfig <path> [-simOut <dir>] [-simVerbose]
    public static void Run()
    {
        string config = GetArg("-simConfig");
        string outDir = GetArg("-simOut");

        if (string.IsNullOrEmpty(config))
        {
            Debug.LogError("[Sim] Missing required argument: -simConfig <path>");
            EditorApplication.Exit(2);
            return;
        }

        GeneratorSession.Begin(config, outDir, quitWhenDone: true, verbose: HasFlag("-simVerbose"));
    }

    // Command-line helpers
    static string GetArg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == name) return args[i + 1];
        return null;
    }

    static bool HasFlag(string name)
    {
        foreach (var a in Environment.GetCommandLineArgs())
            if (a == name) return true;
        return false;
    }
}
