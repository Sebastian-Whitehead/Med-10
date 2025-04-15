using System;
using UnityEngine;

public class SimplifyMesh : MonoBehaviour
{
    public float currentQuality = 0.5f;
    public MeshFilter meshFilter;
    public int triCount = 0;

    private bool useSigmoid = true;

    void Awake()
    {
        meshFilter = this.GetComponent<MeshFilter>();
        triCount = meshFilter.sharedMesh.triangles.Length / 3;
    }

    void Start()
    {   
        if (meshFilter == null)
        {
            Debug.LogError($"MeshFilter is missing on this GameObject: {this.gameObject.name}");
            return;
        }
        // Find the single instance of LightingLevelControl in the scene
        HDRPAssetSwitcher lightingControl = FindObjectOfType<HDRPAssetSwitcher>();
        useSigmoid = lightingControl.useSigmoid;

        if (lightingControl != null)
        {
            // Use the currentDecimateStrength value to determine the base quality
            float baseQuality = Mathf.Clamp01(lightingControl.currentDecimateStrength);
            
            if (baseQuality == 1f) {
                currentQuality = 1f;
                return; // No decimation needed if quality is 1
            }

            // Adjust quality based on mesh complexity
            float adjustedQuality = AdjustQualityBasedOnComplexity(baseQuality);
            //adjustedQuality = Mathf.Max(adjustedQuality, minimumQuality);
            
            decimate(adjustedQuality);
        }
        else
        {
            Debug.LogError("LightingLevelControl instance not found in the scene.");
        }
    }

    private float AdjustQualityBasedOnComplexity(float baseQuality)
    {
        var mesh = meshFilter.sharedMesh;
        if (mesh == null)
        {
            Debug.LogError($"MeshFilter or sharedMesh is missing on {this.gameObject.name}");
            return baseQuality;
        }

        if (!useSigmoid) { // if sigmoid is not to be used, return the base quality
            return baseQuality;
        }

        // Use the triangle count as a measure of complexity
        int triangleCount = mesh.triangles.Length / 3;

        // Log the starting complexity of the model
        //Debug.Log($"Model '{this.gameObject.name}' has {triangleCount} triangles.");

        // Get the dynamic threshold from the MeshManager
        float threshold = MeshManager.Instance.GetComplexityThreshold();

        float k = 1.2f;
        float c = ((float)triangleCount / threshold) / baseQuality;
        float x0 = 4f;

        float adjustedQuality = 1f / (1f + Mathf.Exp(k * (c - x0)));

        

        return adjustedQuality;
    }

    public void decimate(float quality)
    {
        currentQuality = quality;
        var originalMesh = meshFilter.sharedMesh;
        var meshSimplifier = new UnityMeshSimplifier.MeshSimplifier();

        meshSimplifier.Initialize(originalMesh);
        meshSimplifier.SimplifyMesh(currentQuality);
        var destmesh = meshSimplifier.ToMesh();
        GetComponent<MeshFilter>().sharedMesh = destmesh;
    }
}