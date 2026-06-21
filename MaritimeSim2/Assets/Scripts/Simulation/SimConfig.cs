using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Plain-data scene description loaded from JSON via Unity's JsonUtility.
// Fields absent from the JSON keep the defaults assigned here.
[Serializable]
public class SimConfig
{
    public VideoConfig    video   = new VideoConfig();
    public ShipConfig[]   ships;
    public CameraConfig[] cameras;

    // Loading
    public static SimConfig Load(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentException("Config path is empty.");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Config file not found: {path}");

        string json = File.ReadAllText(path);
        SimConfig cfg;
        try
        {
            cfg = JsonUtility.FromJson<SimConfig>(json);
        }
        catch (Exception e)
        {
            throw new Exception($"Failed to parse config '{path}': {e.Message}", e);
        }

        if (cfg == null)
            throw new Exception($"Config '{path}' parsed to null (is it valid JSON?).");

        cfg.Validate();
        return cfg;
    }

    // Validation
    public void Validate()
    {
        var errors = new List<string>();

        if (video == null) video = new VideoConfig();
        if (video.width  <= 0) errors.Add("video.width must be > 0");
        if (video.height <= 0) errors.Add("video.height must be > 0");
        if (video.fps    <= 0) errors.Add("video.fps must be > 0");
        if (video.duration <= 0f) errors.Add("video.duration must be > 0");

        // H.264/MP4 requires even dimensions; round up and warn rather than fail.
        if (video.width % 2 != 0)
        {
            Debug.LogWarning($"[SimConfig] video.width {video.width} is odd; rounding up to {video.width + 1} (MP4 needs even dimensions).");
            video.width += 1;
        }
        if (video.height % 2 != 0)
        {
            Debug.LogWarning($"[SimConfig] video.height {video.height} is odd; rounding up to {video.height + 1} (MP4 needs even dimensions).");
            video.height += 1;
        }

        if (ships == null || ships.Length == 0)
            errors.Add("at least one ship is required");
        else
            for (int i = 0; i < ships.Length; i++)
            {
                var s = ships[i];
                if (string.IsNullOrEmpty(s.prefab))
                    errors.Add($"ships[{i}].prefab is required");
                if (s.position == null || s.position.Length < 2)
                    errors.Add($"ships[{i}].position needs at least [x, z]");
                if (string.IsNullOrEmpty(s.name))
                    s.name = $"{(string.IsNullOrEmpty(s.prefab) ? "ship" : s.prefab)}_{i}";
            }

        if (cameras == null || cameras.Length == 0)
            errors.Add("at least one camera is required");
        else
        {
            var seen = new HashSet<string>();
            for (int i = 0; i < cameras.Length; i++)
            {
                var c = cameras[i];
                if (string.IsNullOrEmpty(c.name)) c.name = $"camera_{i}";
                if (!seen.Add(c.name))
                    errors.Add($"camera name '{c.name}' is duplicated (names become output file names and must be unique)");
                if (c.position == null || c.position.Length < 3)
                    errors.Add($"cameras[{i}].position needs [x, y, z]");
            }
        }

        if (errors.Count > 0)
            throw new Exception("Invalid config:\n  - " + string.Join("\n  - ", errors));
    }

    public int TotalFrames => Mathf.Max(1, Mathf.RoundToInt(video.duration * video.fps));
}

[Serializable]
public class VideoConfig
{
    public int   width    = 1920;
    public int   height   = 1080;
    public int   fps      = 30;
    public float duration = 20f;
}

[Serializable]
public class ShipConfig
{
    public string  name;
    public string  prefab;         // prefab asset name in Assets/ShipsPrefabs
    public float[] position;       // [x, z] or [x, y, z] world space
    public float   heading;        // degrees, yaw (0 = +Z, 90 = +X)
    public float   speed;          // metres / second (exact)
    public int     classId = 8;    // MOT "class" column for this vessel

    public Vector3 Position()
    {
        if (position == null || position.Length == 0) return Vector3.zero;
        if (position.Length == 2) return new Vector3(position[0], 0f, position[1]);
        return new Vector3(position[0], position[1], position[2]);
    }
}

[Serializable]
public class CameraConfig
{
    public string  name;
    public float[] position;       // [x, y, z] world space
    public float[] rotation;       // [pitch, yaw, roll] euler degrees
    public float   fov = 60f;      // vertical field of view

    public Vector3 Position()
    {
        if (position == null || position.Length < 3) return Vector3.zero;
        return new Vector3(position[0], position[1], position[2]);
    }

    public Quaternion Rotation()
    {
        if (rotation == null || rotation.Length < 3) return Quaternion.identity;
        return Quaternion.Euler(rotation[0], rotation[1], rotation[2]);
    }
}
