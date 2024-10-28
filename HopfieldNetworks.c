#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

#define THRESHOLD 0.0

typedef struct {
    int neurons;
    float **weights;
} HopfieldNetwork;

HopfieldNetwork createHopfieldNetwork(int neurons) {
    HopfieldNetwork network;
    network.neurons = neurons;
    network.weights = (float **)malloc(neurons * sizeof(float *));
    for (int i = 0; i < neurons; i++) {
        network.weights[i] = (float *)malloc(neurons * sizeof(float));
        for (int j = 0; j < neurons; j++) {
            network.weights[i][j] = 0.0;
        }
    }
    return network;
}

int main(int argc, char const *argv[]) {
    (void)argc, (void)argv;
    Dataset trainData = readDataset("data/train.txt", TRAIN);
    Dataset testData = readDataset("data/test.txt", TEST);
    assert (trainData.features == testData.features && "Number of features in the training and testing datasets should be the same");
    int features = trainData.features;
    printf("Number of features: %d\n", features);
    printf("Number of instances in the train set: %d\n", trainData.instances);
    printf("Number of instances in the test set: %d\n", testData.instances);

    // Initialize the Hopfield network
    HopfieldNetwork network = createHopfieldNetwork(features);

    // Initialize the state of the network
    int *state = malloc(features * sizeof(int));
    for (int i = 0; i < features; i++) {
        // state[i] = trainData.input[0][i];
        state[i] = rand() % 2;
        // state[i] = 0;
    }

    // Print the initial state of the network
    printf("Initial state of the network:\n");
    for (int i = 0; i < features; i++) {
        printf("%d ", state[i]);
    }
    printf("\n");

    // Update the weights of the Hopfield network
    for (int inst = 0; inst < trainData.instances; inst++) {
        // Update the weights of the network using Hebb's rule
        for (int i = 0; i < features; i++) {
            for (int j = 0; j < features; j++) {
                network.weights[i][j] += (2*state[i] - 1) * (2*state[j] - 1);
                network.weights[j][i] = network.weights[i][j];  // Ensure symmetry
            }
            // Update the diagonal elements of the weight matrix to 0
            network.weights[i][i] = 0.0;
        }
    }
    // Normalize the weights of the network
    for (int i = 0; i < features; i++) {
        for (int j = 0; j < features; j++) {
            network.weights[i][j] /= features;
        }
    }

    // Update the state of the network using the input pattern
    for (int i = 0; i < features; i++) {
        float sum = 0.0;
        for (int j = 0; j < features; j++) {
            sum += network.weights[i][j] * state[j];
        }
        state[i] = sum > THRESHOLD ? 1 : 0;
    }
    // compute the energy of the network
    float energy = 0.0;
    for (int i = 0; i < features; i++) {
        for (int j = 0; j < features; j++) {
            energy += network.weights[i][j] * state[i] * state[j];
        }
    }
    energy *= -0.5;
    // for (int i = 0; i < features; i++) {
    //     energy -= state[i] * THRESHOLD;
    // }
    #ifdef DEBUG
        printf("Energy of the network: %.2f\n", energy);
    #endif
    
    // Print the final state of the network
    printf("Final state of the network (after training):\n");
    for (int i = 0; i < features; i++) {
        printf("%d ", state[i]);
    }
    printf("\n");

    // Test the network using the test dataset
    int correct_predictions = 0;
    for (int test_inst = 0; test_inst < testData.instances; test_inst++) {
        // Initialize the network state to the current test pattern
        for (int i = 0; i < features; i++) {
            state[i] = testData.input[test_inst][i];  // 0 and 1 as binary states
        }

        // Run the network until convergence
        int stable;
        do {
            stable = 1;
            for (int i = 0; i < features; i++) {
                float sum = 0.0;
                for (int j = 0; j < features; j++) {
                    sum += network.weights[i][j] * state[j];
                }
                int new_state = sum > THRESHOLD ? 1 : 0;
                if (new_state != state[i]) {
                    stable = 0;  // Not yet converged
                    state[i] = new_state;
                }
            }
        } while (!stable);  // Repeat until all neurons are stable

        // Check if the final state matches the target pattern for this test instance
        int match = 1;
        for (int i = 0; i < features; i++) {
            if (state[i] != testData.output[test_inst]) {
                match = 0;
                break;
            }
        }
        if (match) correct_predictions++;
    }

    // Calculate accuracy
    float accuracy = ((float)correct_predictions / testData.instances) * 100.0;
    printf("Accuracy on test set: %.2f%%\n", accuracy);


    // Free the memory allocated for the input and output arrays + state
    freeDataset(trainData);
    freeDataset(testData);
    free(state);
    return 0;
}
