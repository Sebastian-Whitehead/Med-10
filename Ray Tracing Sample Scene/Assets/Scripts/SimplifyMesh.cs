using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimplifyMesh : MonoBehaviour
{
    public float currentQuality = 0.5f;

    // Start is called before the first frame update
    void Start()
    {
        // Find the single instance of LightingLevelControl in the scene
        HDRPAssetSwitcher lightingControl = FindObjectOfType<HDRPAssetSwitcher>();
        if (lightingControl != null)
        {
            // Use the currentDecimateStrength value to determine the quality
            float quality = Mathf.Clamp01(lightingControl.currentDecimateStrength);
            decimate(quality);
        }
        else
        {
            Debug.LogError("LightingLevelControl instance not found in the scene.");
        }
    }

    public void decimate(float quality)
    {
        currentQuality = quality;
        var originalMesh = GetComponent<MeshFilter>().sharedMesh;
        var meshSimplifier = new UnityMeshSimplifier.MeshSimplifier();

        meshSimplifier.Initialize(originalMesh);
        meshSimplifier.SimplifyMesh(currentQuality);
        var destmesh = meshSimplifier.ToMesh();
        GetComponent<MeshFilter>().sharedMesh = destmesh;
    }
}
