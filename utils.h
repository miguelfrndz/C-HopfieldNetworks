#pragma once
#ifndef UTILS_HOPFIELD_H
#define UTILS_HOPFIELD_H

typedef enum {
    TRAIN,
    TEST
} SplitType;

typedef struct {
    int instances;
    int features;
    float **input;
    int *output;
    SplitType split;
} Dataset;

extern Dataset readDataset(char *filename, SplitType split);
extern void freeDataset(Dataset dataset);

#endif