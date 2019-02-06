//
// Created by André Schnabel on 2019-02-06.
//

#include <cmath>
#include <iostream>
#include "Bagging.hpp"
#include "Utils.h"

std::vector<int> randomIndices(int count, int n) {
	std::vector<int> res(static_cast<unsigned long>(count));
	int i=0;
	bool alreadyChosen;
	while(i < count) {
		int ix = Utils::randRangeIncl(0, n-1);
		alreadyChosen = false;
		for(int j=0; j<i; j++) {
			if(res[j] == ix) {
				alreadyChosen = true;
				break;
			}
		}
		if(!alreadyChosen) {
			res[i] = ix;
			i++;
		}
	}
	return res;
}

Matrix<float> extractSubSample(const Matrix<float> &instances, const std::vector<int> &selectedInstances, const std::vector<int> &selectedFeatures) {
	return Utils::constructVector<std::vector<float>>(static_cast<int>(selectedInstances.size()), [&selectedFeatures, &instances, &selectedInstances](int i) {
		return  Utils::constructVector<float>(static_cast<int>(selectedFeatures.size()), [&instances, &selectedFeatures, &selectedInstances, i](int j) {
			return instances(selectedInstances[i], selectedFeatures[j]);
		});
	});
}

std::list<Node *> createEnsemble(int numTrees, const Matrix<float> &instances, int numFeatures, int subsampleSize) {
	std::list<Node *> ensemble;
	int instanceCount = instances.getM();
	int featureCount = instances.getN()-1;
	for(int i=0; i<numTrees; i++) {
		std::vector<int> selectedFeatures = randomIndices(numFeatures, featureCount);
		selectedFeatures.push_back(featureCount);
		std::vector<int> selectedInstances = randomIndices(subsampleSize, instanceCount);
		Matrix<float> subsample = extractSubSample(instances, selectedInstances, selectedFeatures);
		ensemble.push_back(buildTree(subsample));
		std::cout << "Added tree nr. " << (i+1) << std::endl;
	}
	return ensemble;
}

int predictWithEnsemble(std::list<Node *> ensemble, const std::vector<float> &instance) {
	int acc = 0;
	for(const auto &rootNode : ensemble) {
		acc += predictWithTree(rootNode, instance);
	}
	return static_cast<int>(round(acc / ensemble.size()));
}

std::vector<int> predictWithEnsemble(std::list<Node *> ensemble, const Matrix<float> &instances) {
	return Utils::constructVector<int>(instances.getM(), [&ensemble, &instances](int i) { return predictWithEnsemble(ensemble, instances.row(i)); });
}
