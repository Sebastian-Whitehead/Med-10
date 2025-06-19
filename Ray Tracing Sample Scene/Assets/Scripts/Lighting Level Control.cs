using UnityEditor.EditorTools;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

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

    public void SetLightingLevel(Lighting level)
    {
        // Switch case for each lighting level
        switch (level)
        {
            case Lighting.Off:
                print("Lighting Level Not Implemented");
                break;
            case Lighting.Custom:
                GraphicsSettings.renderPipelineAsset = hdrpAssets[CustomAssetIndex];
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
