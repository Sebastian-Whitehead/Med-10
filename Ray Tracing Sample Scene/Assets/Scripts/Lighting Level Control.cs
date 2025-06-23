using UnityEditor.EditorTools;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.Perception.GroundTruth;
using System.Runtime.CompilerServices;
using GLTFast;
using UnityEngine.SceneManagement;
using Unity.VisualScripting;

public class LightingLevelController : MonoBehaviour
{
    public enum Lighting
    {
        Off,
        Custom,
        Low,
        Medium,
        High,
        Raytracing,
        Pathtracing
    }
    [Header("Lighting Quality Settings")]
    [Tooltip("Custom Index of HDRP Assets in the Graphics settings menu.")]
    public int customHDRPAssetsIndex = 0;

    [Tooltip("If a \"Custom\" HDRP Asset is selected, this index will be used to determine which asset from the customHDRPAssets Array to apply.")]
    public int CustomAssetIndex = 0;
    public Volume PtVolume;
    public PerceptionCamera perceptionCamera;

    public GameObject[] lightingControlObjects;

    public void SetLightingLevel(Lighting level)
    {
        bool PT_Enabled = false;
        int pts = 0;
        Debug.Log($"Setting lighting level to: {level}");
        switch (level)
        {
            case Lighting.Off:
                foreach (GameObject obj in lightingControlObjects)
                {
                    obj.SetActive(false);
                }
                print("Lighting is turned off.");
                break;
            case Lighting.Custom:
                QualitySettings.SetQualityLevel(customHDRPAssetsIndex, true);
                break;
            case Lighting.Low:
                QualitySettings.SetQualityLevel(0, true);
                break;
            case Lighting.Medium:
                QualitySettings.SetQualityLevel(1, true);
                break;
            case Lighting.High:
                QualitySettings.SetQualityLevel(2, true);
                break;
            case Lighting.Raytracing:
                QualitySettings.SetQualityLevel(3, true);
                break;
            case Lighting.Pathtracing:
                QualitySettings.SetQualityLevel(4, true);
                PtVolume.profile.TryGet(out PathTracing pathTracingVolume);
                pts = FindObjectOfType<VariableControl>().PathtracingSamples;
                pathTracingVolume.maximumSamples.Override(pts);
                PT_Enabled = true;
                break;
        }
        PtVolume.gameObject.SetActive(PT_Enabled);
        perceptionCamera.useAccumulation = PT_Enabled;
        FindObjectOfType<ItemRandomizer>().SetPathTracingSamples(pts, PT_Enabled);

        //Debug.Log($"Switched HDRP Asset to: {GraphicsSettings.currentRenderPipeline.name}");
    }

    public int GetAccumulationSamples()
    {
        // Access the active volume stack
        var volumeStack = VolumeManager.instance.stack;

        // Retrieve the PathTracing component from the volume stack
        PathTracing pathTracing = volumeStack.GetComponent<PathTracing>();
        if (pathTracing != null && pathTracing.active)
        {
            return pathTracing.maximumSamples.value; // Access the maximum samples value
        }

        Debug.LogWarning("PathTracing component not found or not active in the volume stack.");
        return 0; // Default value if not found
    }
}
