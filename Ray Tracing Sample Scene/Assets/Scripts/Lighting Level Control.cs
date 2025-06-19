using UnityEditor.EditorTools;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.Perception.GroundTruth;
using System.Runtime.CompilerServices;

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

    [Tooltip("Do Not Modify this array")]
    public HDRenderPipelineAsset[] hdrpAssets; // Assign different HDRP Assets in the Inspector

    [Tooltip("Custom HDRP Assets for different lighting levels. ")]
    public HDRenderPipelineAsset[] CustomHDRPAssets; // Custom HDRP Assets for different lighting levels

    [Tooltip("If a \"Custom\" HDRP Asset is selected, this index will be used to determine which asset from the customHDRPAssets Array to apply.")]
    public int CustomAssetIndex = 0;
    public Volume PtVolume;
    public PerceptionCamera perceptionCamera;

    public void SetLightingLevel(Lighting level)
    {
        bool PT_Enabled = false;
        int pts = 0;

        switch (level)
        {
            case Lighting.Off:
                print("Lighting Level Not Implemented");
                break;
            case Lighting.Custom:
                GraphicsSettings.renderPipelineAsset = CustomHDRPAssets[CustomAssetIndex];
                break;
            case Lighting.Low:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[3];
                break;
            case Lighting.Medium:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[2];
                break;
            case Lighting.High:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[1];
                break;
            case Lighting.Raytracing:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[0];
                break;
            case Lighting.Pathtracing:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[0];
                PtVolume.profile.TryGet(out PathTracing pathTracingVolume);
                pts = FindObjectOfType<VariableControl>().PathtracingSamples;
                pathTracingVolume.maximumSamples.Override(pts);
                PT_Enabled = true;
                break;
        }
        PtVolume.gameObject.SetActive(PT_Enabled);
        perceptionCamera.useAccumulation = PT_Enabled;
        FindObjectOfType<ItemRandomizer>().SetPathTracingSamples(pts, PT_Enabled);

        Debug.Log($"Switched HDRP Asset to: {GraphicsSettings.currentRenderPipeline.name}");
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
