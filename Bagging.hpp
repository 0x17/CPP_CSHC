//
// Created by André Schnabel on 2019-02-06.
//

#pragma once

#include <list>

#include "Matrix.h"
#include "Cart.hpp"

std::list<Node *> createEnsemble(int numTrees, const Matrix<float> &instances, int numFeatures, int subsampleSize);
int predictWithEnsemble(std::list<Node *> ensemble, const std::vector<float> &instance);
std::vector<int> predictWithEnsemble(std::list<Node *> ensemble, const Matrix<float> &instances);