#pragma once

#include "Matrix.h"

struct SplitChoice {
	int featureIndex;
	float value;

	explicit SplitChoice(int featureIndex = 0, float value = 0.0f) : featureIndex(featureIndex), value(value) {}
};

struct SplitResult : public SplitChoice {
	int cost;
	Matrix<float> left, right;

	SplitResult(int featureIndex, float value, int cost, const Matrix<float> &left, const Matrix<float> &right)
	: SplitChoice(featureIndex, value), cost(cost), left(left), right(right) {}
};

struct Node {
	SplitChoice split;
	Node *left, *right;

	Node(int _featureIndex, float _value, int _cost, Node *_left, Node *_right) : split(_featureIndex, _value), left(_left), right(_right) {}
	~Node() {
		delete left;
		delete right;
	}
};

Node* buildTree(const Matrix<float> &instances);

int predictWithTree(const Node *root, const std::vector<float> &instance);
std::vector<int> predictWithTree(const Node *root, const Matrix<float> &instances);