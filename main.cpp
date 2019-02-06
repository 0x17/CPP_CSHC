
#include <iostream>
#include <cmath>
#include "Matrix.h"
#include "Utils.h"
#include "Cart.hpp"
#include "Bagging.hpp"

int main() {
	const bool useEnsemble = false;

	const auto instances = Utils::parseCsvValues("dataforcshc.csv");

	const auto res = Utils::trainValidationSplit(instances, 0.9, 23, true);
	const auto train = res.first;
	const auto validation = res.second;

	int classIndex = validation.getN()-1;
	std::vector<int> actualYs = Utils::constructVector<int>(validation.getM(), [classIndex,&validation](int i) { return validation(i,classIndex); });

	std::vector<int> predYs;

	if(useEnsemble) {
		const int numFeatures = static_cast<const int>(round(sqrt(train.getN() - 1.0f) * 0.7f));
		const int subSampleSize = static_cast<const int>(round(0.7f * train.getM()));
		const auto ensemble = createEnsemble(8, instances, numFeatures, subSampleSize);
		predYs = predictWithEnsemble(ensemble, validation);
		for(Node *root : ensemble) {
			delete root;
		}
	} else {
		Node *root = buildTree(train);
		predYs = predictWithTree(root, validation);
		delete root;
	}

	std::cout << "Prediction accuracy = " << Utils::accuracy(actualYs, predYs) << std::endl;
}
