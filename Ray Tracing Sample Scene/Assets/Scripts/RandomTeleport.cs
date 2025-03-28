using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomTeleport : MonoBehaviour
{
    [Header("Teleport Settings")]
    public List<GameObject> positionalReferences = new List<GameObject>(); // List of empty GameObjects as positional references
    public GameObject targetObject; // The object to teleport


    void Start()
    {
        Teleport();
    }
    /// <summary>
    /// Teleports the target object to a random position from the positional references.
    /// </summary>
    public void Teleport()
    {
        if (positionalReferences.Count == 0)
        {
            Debug.LogWarning("No positional references assigned for teleportation.");
            return;
        }

        if (targetObject == null)
        {
            Debug.LogWarning("No target object assigned for teleportation.");
            return;
        }

        // Select a random position from the list
        int randomIndex = Random.Range(0, positionalReferences.Count);
        GameObject randomReference = positionalReferences[randomIndex];

        // Move the target object to the selected position
        targetObject.transform.position = randomReference.transform.position;

        Debug.Log($"Teleported to {randomReference.name}'s position.");
    }
}

