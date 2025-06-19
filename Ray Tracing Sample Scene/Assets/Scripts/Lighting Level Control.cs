using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class HDRPAssetSwitcher : MonoBehaviour
{
    public enum Lighting
    {
        Off,
        Low,
        Medium,
        High,
        Raytracing,
        Pathtracing
    }

    public HDRenderPipelineAsset[] hdrpAssets; // Assign different HDRP Assets in the Inspector
    public void SetLightingLevel(Lighting level)
    {
        // Switch case for each lighting level
        switch (level)
        {
            case Lighting.Off:
                print("Lighting Level Not Implemented");
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
                print("Lighting Level Not Implemented");
                break;
        }

        Debug.Log($"Switched HDRP Asset to: {GraphicsSettings.currentRenderPipeline.name}");
    }
}
