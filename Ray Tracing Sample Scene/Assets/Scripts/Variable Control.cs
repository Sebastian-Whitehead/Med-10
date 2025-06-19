using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class VariableControl : MonoBehaviour
{

    [Header("Lighting Quality Settings")]
    [Tooltip("The current lighting level for the scene.")]
    public HDRPAssetSwitcher.Lighting lightingLevel = HDRPAssetSwitcher.Lighting.Medium;

    [Tooltip("Number of samples used for path tracing. Only applicable if lightingLevel is set to Pathtracing.")]
    public int PathtracingSamples = 256;

    [Header("Polygon Decimation Settings")]
    [Tooltip("Strength of the decimation applied to the scene.")]
    [Range(0f, 1f)]
    public float DecimationStrength = 0.1f;

    [Tooltip("Applies a sigmoid function to the decimation strength to equalize the poly crush effect across models")]
    public bool useSigmoid = true;

    [Header("Texture Quality Settings")]
    [Tooltip("Quality level of textures (0 = lowest, higher values = better quality).")]
    [Min(0)]
    public int TextureQuality = 0;


    [Header("General Simulator Settings")]
    [Tooltip("Maximum number of captures allowed.")]
    public int CaptureLimit = 200;

    [ReadOnly]
    [Tooltip("Current number of captures (read-only).")]
    public int CaptureCount = 0;

    private float currentDecimateStrength = 1.0f;

    private ItemRandomizer itemRandomizer;

    // Start is called before the first frame update
    void Start()
    {
        
        itemRandomizer = FindObjectOfType<ItemRandomizer>();
        itemRandomizer.captureLimit = CaptureLimit;

        // Set initial lighting level
        HDRPAssetSwitcher lightingControler = this.GetComponent<HDRPAssetSwitcher>();
        lightingControler.SetLightingLevel(lightingLevel);
        
        // Set initial texture quality
        SetTextureQuality(TextureQuality);
        
        // Set initial decimation strength
        SetCurrentDecimateStrength(DecimationStrength);
    }

    // Update is called once per frame
    void Update()
    {
        CaptureCount = itemRandomizer.captureCount;
    }

    private void SetTextureQuality(int mipMap)
    {
        QualitySettings.globalTextureMipmapLimit = mipMap;
    }
    
    public void SetCurrentDecimateStrength(float strength)
    {
        currentDecimateStrength = strength;
    }
}
