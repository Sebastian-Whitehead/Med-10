using System.Collections;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class PathTracingHandler : MonoBehaviour
{
    public PerceptionCamera perceptionCamera; // Reference to the PerceptionCamera
    public bool PathTracingEnabled = true;   // Flag to enable or disable path tracing
    public Camera accumulationCamera;
    public System.Action placeholderFunction; // Function to call after rendering

    public bool isRendering = false;
    private int iteration = 0;
    public int samples = 128;  // Number of accumulated samples
    private bool waitForNextFrame = false; // Flag to wait for external trigger

    private void Start()
    {
        if (accumulationCamera == null)
        {
            accumulationCamera = Camera.main;
        }
        RenderPipelineManager.beginFrameRendering += OnBeginFrameRendering;
    }

    private void OnDestroy()
    {
        RenderPipelineManager.beginFrameRendering -= OnBeginFrameRendering;
    }

    public void StartCapture()
    {
        if (isRendering) return; // Prevent re-entry
        if (!PathTracingEnabled)
        {
            perceptionCamera?.RequestCapture();
            isRendering = true; // Mark frame as complete for non-path-tracing mode
            return;
        }

        StartCoroutine(WaitForAccumulation());
    }

    private IEnumerator WaitForAccumulation()
    {
        isRendering = true;
        iteration = 0;

        HDRenderPipeline renderPipeline = RenderPipelineManager.currentPipeline as HDRenderPipeline;
        if (renderPipeline != null)
        {
            renderPipeline.BeginRecording(samples, 1.0f, 0.25f, 0.75f); // Adjust parameters as needed
        }

        while (iteration < samples)
        {
            yield return null; // Wait for each sub-frame to render
        }

        renderPipeline?.EndRecording();
        

        // Request a capture from the PerceptionCamera after accumulation
        perceptionCamera?.RequestCapture();

        yield return new WaitForSeconds(5f); // Adjust the delay duration as needed
        isRendering = false;
    }

    private void OnBeginFrameRendering(ScriptableRenderContext context, Camera[] cameras)
    {
        if (!isRendering || waitForNextFrame) return;

        HDRenderPipeline renderPipeline = RenderPipelineManager.currentPipeline as HDRenderPipeline;
        renderPipeline?.PrepareNewSubFrame();
        iteration++;
    }

    /// <summary>
    /// Call this method to resume rendering after objects have settled.
    /// </summary>
    public void ResumeRendering()
    {
        if (!waitForNextFrame) return;

        waitForNextFrame = false;
        Debug.Log("Resuming rendering...");
        StartCapture();
    }
}