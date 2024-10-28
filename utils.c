#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "utils.h"

Dataset readDataset(char *filename, SplitType split) {
    if (split == TRAIN) {
        printf("Loading training dataset from file '%s'\n", filename);
    } else if (split == TEST) {
        printf("Loading testing dataset from file '%s'\n", filename);
    } else{
        fprintf(stderr, "Invalid split type\n");
        exit(1);
    }
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        // perror("Error opening file");
        fprintf(stderr, "Error opening file '%s' -- %s\n", filename, strerror(errno));
        exit(1);
    }

    Dataset dataset;
    // Specify the split type (i.e, TRAIN or TEST)
    dataset.split = split;
    // Read the number of instances and features
    fscanf(file, "%d", &dataset.instances);
    fscanf(file, "%d", &dataset.features);

    // Allocate memory for the input and output arrays
    dataset.input = (int **)malloc(dataset.instances * sizeof(int *));
    dataset.output = (int *)malloc(dataset.instances * sizeof(int));
    for (int i = 0; i < dataset.instances; i++) {
        dataset.input[i] = (int *)malloc(dataset.features * sizeof(int));
        for (int j = 0; j < dataset.features; j++) {
            fscanf(file, "%d", &dataset.input[i][j]);
        }
        // Add label of the instance to the output array
        fscanf(file, "%d", &dataset.output[i]);
    }

    fclose(file);
    return dataset;
}

void freeDataset(Dataset dataset) {
    for (int i = 0; i < dataset.instances; i++) {
        free(dataset.input[i]);
    }
    free(dataset.input);
    free(dataset.output);
}