#include <cmath>
#include <iostream>
#include "Cart.hpp"

int splitCost(const Matrix<float> &instances, int featureIndex, float splitValue) {
	int sleft = 0, sright = 0, sleftCard = 0, srightCard = 0;
	int classIndex = instances.getN()-1;
	for(int i=0; i<instances.getM(); i++) {
		if(instances(i,featureIndex) < splitValue) {
			sleft += (int)instances(i,classIndex);
			sleftCard++;
		} else {
			sright += (int)instances(i,classIndex);
			srightCard++;
		}
	}
	int nmisclassLeft = sleft > (int)ceil((float)sleftCard / 2.0f) ? sleftCard-sleft : sleft;
	int nmisclassRight = sright > (int)ceil((float)srightCard / 2.0f) ? srightCard-sright : sright;
	return nmisclassLeft+nmisclassRight;
}

Matrix<float> leftPart(const Matrix<float> &instances, const SplitChoice &sc) {
	std::vector<std::vector<float>> outInstances;
	for(int i=0; i<instances.getM(); i++) {
		if(instances(i,sc.featureIndex) < sc.value) {
			outInstances.push_back(instances.row(i));
		}
	}
	return outInstances;
}

Matrix<float> rightPart(const Matrix<float> &instances, const SplitChoice &sc) {
	std::vector<std::vector<float>> outInstances;
	for(int i=0; i<instances.getM(); i++) {
		if(instances(i,sc.featureIndex) >= sc.value) {
			outInstances.push_back(instances.row(i));
		}
	}
	return outInstances;
}

SplitResult cheapestSplit(const Matrix<float> &instances) {
	SplitChoice sc;
	int bestCost = std::numeric_limits<int>::max();
	int nfeatures = instances.getN()-1;

	for(int featureIndex = 0; featureIndex<nfeatures; featureIndex++) {
		for(int i = 0; i<instances.getM(); i++) {
			float v = instances(featureIndex, i);
			int cost = splitCost(instances, featureIndex, v);
			if(cost < bestCost) {
				bestCost = cost;
				sc.featureIndex = featureIndex;
				sc.value = v;
				std::cout << "New incumbent with cost " << cost << std::endl;
			}
		}
	}

	std::cout << "New best split feature=" << sc.featureIndex << " value=" << sc.value << std::endl;
	return SplitResult(sc.featureIndex, sc.value, bestCost, leftPart(instances, sc), rightPart(instances, sc));
}

int dominatingClass(const Matrix<float>& instances) {
	const int lastIx = instances.getN() - 1;
	int acc = 0;
	for (int i = 0; i < instances.getM(); i++)
		acc += (int)instances(i, lastIx);
	return acc > static_cast<int>(round(static_cast<double>(instances.getM()) / 2.0)) ? 1 : 0;
}

Node *buildTree(const Matrix<float>& instances) {
	const auto terminalNode = [](int v, int cost) {
		return new Node(-1, v, cost, nullptr, nullptr);
	};

	SplitResult s = cheapestSplit(instances);
	if(s.cost == 0.0) {
		int dcl = dominatingClass(s.left);
		int dcr = dominatingClass(s.right);
		if (dcl == dcr) return terminalNode(dcl, s.cost);
		return new Node(s.featureIndex, s.value, s.cost, terminalNode(dcl, s.cost), terminalNode(dcr, s.cost));
	}

	int lsize = s.left.getM();
	int rsize = s.right.getM();

	if(lsize == 0) return terminalNode(dominatingClass(s.left), s.cost);
	if(rsize == 0) return terminalNode(dominatingClass(s.right), s.cost);

	return new Node(s.featureIndex, s.value, s.cost, buildTree(s.left), buildTree(s.right));
}

int predictWithTree(const Node *root, const std::vector<float> &instance) {
	if(instance[root->split.featureIndex] < root->split.value) {
		if(!root->left) {
			return (int)root->split.value;
		} else {
			return predictWithTree(root->left, instance);
		}
	} else {
		if(!root->right) {
			return (int)root->split.value;
		} else {
			return predictWithTree(root->right, instance);
		}
	}
}
