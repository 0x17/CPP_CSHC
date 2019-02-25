#include <cmath>
#include <iostream>
#include "Cart.hpp"
#include "Utils.h"

int splitCost(const Matrix<float> &instances, int featureIndex, float splitValue) {
	int sleft = 0, sright = 0, sleftCard = 0, srightCard = 0;
	const int classIndex = instances.getN()-1;
	for(int i=0; i<instances.getM(); i++) {
		if(instances(i,featureIndex) < splitValue) {
			sleft += static_cast<int>(instances(i, classIndex));
			sleftCard++;
		} else {
			sright += static_cast<int>(instances(i, classIndex));
			srightCard++;
		}
	}
	const int nmisclassLeft = sleft > (int)ceil((float)sleftCard / 2.0f) ? sleftCard-sleft : sleft;
	const int nmisclassRight = sright > (int)ceil((float)srightCard / 2.0f) ? srightCard-sright : sright;
	return nmisclassLeft + nmisclassRight;
}

SplitResult cheapestSplit(const Matrix<float> &instances) {
	SplitChoice sc;
	int bestCost = std::numeric_limits<int>::max();
	const int nfeatures = instances.getN()-1;

	for(int featureIndex = 0; featureIndex<nfeatures; featureIndex++) {
		for(int i = 0; i<instances.getM(); i++) {
			const float v = instances(featureIndex, i);
			const int cost = splitCost(instances, featureIndex, v);
			if(cost <= bestCost) {
				bestCost = cost;
				sc.featureIndex = featureIndex;
				sc.value = v;
				//std::cout << "New incumbent with cost " << cost << std::endl;
			}
		}
	}

	//std::cout << "New best split feature=" << sc.featureIndex << " value=" << sc.value << std::endl;

	const auto leftPart = instances.filter([&sc](const std::vector<float> &row) {
		return row[sc.featureIndex] < sc.value;
	});

	const auto rightPart = instances.filter([&sc](const std::vector<float> &row) {
		return row[sc.featureIndex] >= sc.value;
	});

	return SplitResult(sc.featureIndex, sc.value, bestCost, leftPart, rightPart);
}

int dominatingClass(const Matrix<float>& instances) {
	const int lastIx = instances.getN() - 1;
	int acc = 0;
	for (int i = 0; i < instances.getM(); i++)
		acc += static_cast<int>(instances(i, lastIx));
	const int threshold = static_cast<int>(round(static_cast<double>(instances.getM()) / 2.0));
	return acc > threshold ? 1 : 0;
}

Node *buildTree(const Matrix<float>& instances) {
	const auto leafNode = [](int v, int cost) {
		return new Node(-1, v, cost, nullptr, nullptr);
	};

	SplitResult s = cheapestSplit(instances);
	if(s.cost == 0) {
		const int dcl = dominatingClass(s.left);
		const int dcr = dominatingClass(s.right);
		if (dcl == dcr) return leafNode(dcl, s.cost);
		return new Node(s.featureIndex, s.value, s.cost, leafNode(dcl, 0), leafNode(dcr, 0));
	}

	const int lsize = s.left.getM();
	const int rsize = s.right.getM();

	if(lsize == 0) return leafNode(dominatingClass(s.right), s.cost);
	if(rsize == 0) return leafNode(dominatingClass(s.left), s.cost);

	return new Node(s.featureIndex, s.value, s.cost, buildTree(s.left), buildTree(s.right));
}

int predictWithTree(const Node *root, const std::vector<float> &instance) {
	if(instance[root->split.featureIndex] < root->split.value) {
		if(!root->left) {
			return static_cast<int>(root->split.value);
		}
		return predictWithTree(root->left, instance);
	}

	if(!root->right) {
		return static_cast<int>(root->split.value);
	}
	return predictWithTree(root->right, instance);
}

std::vector<int> predictWithTree(const Node *root, const Matrix<float> &instances) {
	return Utils::constructVector<int>(instances.getM(), [&root,&instances](int i) {
		return predictWithTree(root, instances.row(i));
	});
}
