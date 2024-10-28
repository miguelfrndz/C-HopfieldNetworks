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
    int stable;
    do {
        stable = 1;
        for (int i = 0; i < features; i++) {
            float sum = 0.0;
            for (int j = 0; j < features; j++) {
                sum += network->weights[i][j] * state[j];
            }
            int new_state = sum > THRESHOLD ? 1 : 0;
            if (new_state != state[i]) {
                stable = 0;
                state[i] = new_state;
            }
        }
    } while (!stable);
}

int main(int argc, char const *argv[]) {
    (void)argc, (void)argv;
    Dataset trainData = readDataset("data/train.txt", TRAIN);
    Dataset testData = readDataset("data/test.txt", TEST);
    assert(trainData.features == testData.features && "Number of features in the training and testing datasets should be the same");
    int features = trainData.features;

    HopfieldNetwork network = createHopfieldNetwork(features);
    trainHopfieldNetwork(&network, trainData);

    int *state = malloc(features * sizeof(int));
    int correct_predictions = 0;
    for (int test_inst = 0; test_inst < testData.instances; test_inst++) {
        for (int i = 0; i < features; i++) {
            state[i] = testData.input[test_inst][i];
        }

        updateState(&network, state);

        int match = 1;
        for (int i = 0; i < features; i++) {
            if (state[i] != testData.output[test_inst]) {
                match = 0;
                break;
            }
        }
        if (match) correct_predictions++;
    }

    float accuracy = ((float)correct_predictions / testData.instances) * 100.0;
    printf("Accuracy on test set: %.2f%%\n", accuracy);

    for (int i = 0; i < features; i++) {
        free(network.weights[i]);
    }
    free(network.weights);
    freeDataset(trainData);
    freeDataset(testData);
    free(state);
    return 0;
}
