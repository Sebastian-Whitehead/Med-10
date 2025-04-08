using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MeshManager : MonoBehaviour
{
    public static MeshManager Instance { get; private set; }

    private float complexityThreshold = 10000f; // Default fallback value
    private bool thresholdCalculated = false;

    private void Awake()
    {
        // Ensure only one instance of MeshManager exists
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }


    public float GetComplexityThreshold()
    {
        if (!thresholdCalculated)
        {
            CalculateSceneComplexityThreshold();
        }
        return complexityThreshold;
    }

    private void CalculateSceneComplexityThreshold()
    {
        // Find all objects with the SimplifyMesh component
        SimplifyMesh[] simplifyMeshObjects = FindObjectsOfType<SimplifyMesh>();
        if (simplifyMeshObjects.Length == 0)
        {
            Debug.LogWarning("No objects with SimplifyMesh component found in the scene.");
            complexityThreshold = 10000f; // Default fallback value
            return;
        }

        int maxTriangles = 0;
        int minTriangles = int.MaxValue;

        string maxObject = string.Empty;
        string minObject = string.Empty;
        
        List<int> triCounts = new List<int>();

        foreach (SimplifyMesh simplifyMesh in simplifyMeshObjects)
        {
            int triangleCount = simplifyMesh.triCount; // Get the triangle count from the SimplifyMesh component
            triCounts.Add(triangleCount);

            if(triangleCount > maxTriangles)
            {
                maxObject = simplifyMesh.gameObject.name;
                maxTriangles = triangleCount;
            }

            if(triangleCount < minTriangles)
            {
                minObject = simplifyMesh.gameObject.name;
                minTriangles = triangleCount;
            }
        }

        // Sum up all triangle counts
        int totalTrianglesCount = triCounts.Sum();
        int medianTriangleCount = triCounts.OrderBy(x => x).Skip(triCounts.Count / 2).FirstOrDefault(); // Calculate the median triangle count
        // Calculate the average triangle count
        int averageTriangleCount = totalTrianglesCount / triCounts.Count;
        // Calculate the maximum triangle count
        int maxTriangleCount = triCounts.Max();
        int minTriangleCount = triCounts.Min();

        print($"Total Models {triCounts.Count}, Average triangles: {averageTriangleCount}, Median Triangles: {medianTriangleCount}, Max triangles: {maxTriangleCount} for {maxObject}, Min triangles: {minTriangleCount} for object: {minObject}");
        //print triCounts as a comma separated string   
        print(string.Join(", ", triCounts));

        // Calculate average triangle count
        //int averageTriangles = totalTriangles / simplifyMeshObjects.Length;
        // Set the complexity threshold (choose average or max based on preference)
        //print($"Average triangles: {averageTriangles}, Max triangles: {maxTriangles} for object: {maxObject}");
        
        
        complexityThreshold = averageTriangleCount; // Or use maxTriangles for max-based scaling
        thresholdCalculated = true;
    }
}