using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ModelChanger : MonoBehaviour
{
    [Header("Model Settings")]
    public GameObject currentModel; // The currently active model
    public Transform modelParent; // Parent transform to hold the model
    public List<GameObject> modelPrefabs; // List of model prefabs to switch between

    /// <summary>
    /// Replaces the current model with a new model from the list.
    /// </summary>
    /// <param name="modelIndex">Index of the new model in the modelPrefabs list.</param>
    public void ChangeModel(int modelIndex)
    {
        if (modelIndex < 0 || modelIndex >= modelPrefabs.Count)
        {
            Debug.LogWarning("Invalid model index. Cannot change model.");
            return;
        }

        // Destroy the current model if it exists
        if (currentModel != null)
        {
            Destroy(currentModel);
        }

        // Instantiate the new model and set it as the current model
        GameObject newModel = Instantiate(modelPrefabs[modelIndex], modelParent);
        currentModel = newModel;
    }


    public void RandomModel()
    {
        int randomIndex = Random.Range(0, modelPrefabs.Count);
        ChangeModel(randomIndex);
    }

    // if m is pressed, change the model to a random one
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.M))
        {
            RandomModel();
        }
    }
}