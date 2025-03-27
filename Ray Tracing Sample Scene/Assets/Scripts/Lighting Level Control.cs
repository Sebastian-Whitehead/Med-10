using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class HDRPAssetSwitcher : MonoBehaviour
{
    public HDRenderPipelineAsset[] hdrpAssets; // Assign different HDRP Assets in the Inspector
    public int currentHDRPAssetIndex = 0;
    public int currentMipMap = 0;
    public float currentDecimateStrength = 1;

    public void SetHDRPAsset(int index)
    {
        if (index < 0 || index >= hdrpAssets.Length || hdrpAssets[index] == null)
        {
            Debug.LogError("Invalid HDRP asset index.");
            return;
        }

        GraphicsSettings.renderPipelineAsset = hdrpAssets[index];
        Debug.Log($"Switched HDRP Asset to: {hdrpAssets[index].name}");
    }

    void Start()
    {
        SetHDRPAsset(currentHDRPAssetIndex);
        QualitySettings.globalTextureMipmapLimit = currentMipMap;
    }
}
