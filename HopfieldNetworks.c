#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
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

void freeHopfieldNetwork(HopfieldNetwork network) {
    for (int i = 0; i < network.neurons; i++) {
        free(network.weights[i]);
    }
    free(network.weights);
}

float computeEnergy(HopfieldNetwork *network, int *state) {
    float energy = 0.0;
    int features = network->neurons;
    for (int i = 0; i < features; i++) {
        for (int j = i + 1; j < features; j++) {  // Avoid double counting
            energy += network->weights[i][j] * state[i] * state[j];
        }
    }
    return -0.5 * energy;
}

void trainHopfieldNetwork(HopfieldNetwork *network, Dataset trainData) {
    int features = network->neurons;
    for (int inst = 0; inst < trainData.instances; inst++) {
        int *state = trainData.input[inst];
        // Print the state at iteration inst
        #ifdef DEBUG
            printf("State at iteration %d: ", inst);
            for (int i = 0; i < features; i++) {
                printf("%d ", state[i]);
            }
            printf("\n");
        #endif
        for (int i = 0; i < features; i++) {
            for (int j = i + 1; j < features; j++) {
                network->weights[i][j] += (2 * state[i] - 1) * (2 * state[j] - 1);
                network->weights[j][i] = network->weights[i][j];  // Ensure symmetry
            }
            // Set diagonal to zero
            network->weights[i][i] = 0.0;
            // Normalize the weights
            for (int j = 0; j < features; j++) {
                network->weights[i][j] /= features;
            }
        }
        // Print the energy of the network after each training instance
        #ifdef DEBUG
            float energy = computeEnergy(network, state);
            printf("Energy after training instance %d: %.2f\n", inst, energy);
        #endif
    }
}

void updateState(HopfieldNetwork *network, int *state) {
    int features = network->neurons;
    // State should be the sign of weights * state
    for (int i = 0; i < features; i++) {
        float activation = 0.0;
        for (int j = 0; j < features; j++) {
            activation += network->weights[i][j] * state[j];
        }
        state[i] = activation > THRESHOLD ? 1 : 0;
    }
}

void evaluateHopfieldNetwork(HopfieldNetwork *network, Dataset testData, Dataset trainData, int sync_iterations) {
    int features = network->neurons;
    int *state = malloc(features * sizeof(int));
    int correct_predictions = 0;
    float energy;
    for (int test_inst = 0; test_inst < testData.instances; test_inst++) {
        // Syncronous update of the network (i.e, update all neurons at the same time until convergence)
        memcpy(state, testData.input[test_inst], features * sizeof(int));
        energy = computeEnergy(network, state);
        for (int iter = 0; iter < sync_iterations; iter++) {
            updateState(network, state);
            float new_energy = computeEnergy(network, state);
            if (new_energy == energy) {
                break;
            }
            energy = new_energy;
        }
        /* 
        Make prediction and compare with ground-truth:
            - If the stable state matches a specific stored pattern, assign the corresponding binary label (either 1 or 0).
            - If no exact match is found, use a similarity threshold (e.g., Hamming distance) to classify the pattern based on the closest stored pattern.
        */
        int match_found = 0;
        int prediction = -1;
        int min_distance = features;
        for (int inst = 0; inst < trainData.instances; inst++) {
            int distance = 0;
            for (int i = 0; i < features; i++) {
                distance += state[i] != trainData.input[inst][i];
            }
            if (distance < min_distance) {
                min_distance = distance;
                prediction = trainData.output[inst];
            }
            if (distance == 0) {
                match_found = 1;
                break;
            }
        }
        #ifdef DEBUG
            // Print if the prediction was made by exact match or proximity by Hamming distance
            if (match_found) {
                printf(">> Test instance %d - Prediction made by exact match\n", test_inst);
            } else {
                printf(">> Test instance %d - Prediction made by proximity\n", test_inst);
            }
            // Print the network prediction and the ground truth
            printf("Test instance %d - Ground Truth: %d, Prediction: %d\n", test_inst, testData.output[test_inst], prediction);
        #else
            // Suppress unused variable warning (only for DEBUG mode)
            (void)match_found;
        #endif

        if (testData.output[test_inst] == prediction) {
            correct_predictions++;
        }
    }
    float accuracy = ((float)correct_predictions / testData.instances) * 100.0;
    printf("Accuracy on test set: %.2f%%\n", accuracy);
    free(state);
}

int main(int argc, char const *argv[]) {
    (void)argc, (void)argv;
    Dataset trainData = readDataset("data/train.txt", TRAIN);
    Dataset testData = readDataset("data/test.txt", TEST);
    assert(trainData.features == testData.features && "Number of features in the training and testing datasets should be the same");
    int features = trainData.features;

    HopfieldNetwork network = createHopfieldNetwork(features);
    trainHopfieldNetwork(&network, trainData);
    int sync_iterations = 100;
    evaluateHopfieldNetwork(&network, testData, trainData, sync_iterations);

    freeHopfieldNetwork(network);
    freeDataset(trainData);
    freeDataset(testData);
    return 0;
}
