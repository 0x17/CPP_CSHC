
#include <iostream>
#include "Matrix.h"
#include "Utils.h"
#include "Cart.hpp"

int main() {
	const auto instances = Utils::parseCsvValues("dataforcshc.csv");

	const auto res = Utils::trainValidationSplit(instances, 0.9, 42, true);
	const auto train = res.first;
	const auto validation = res.second;

	Node *root = buildTree(train);
	/*int predY = predictWithTree(root, instances.row(0));
	std::cout << "Predicted class = " << predY << std::endl;*/

	const auto predYs = predictWithTree(root, validation);
	int classIndex = validation.getN()-1;
	std::vector<int> actualYs = Utils::constructVector<int>(validation.getM(), [classIndex,&validation](int i) { return validation(i,classIndex); });

	std::cout << "Prediction accuracy = " << Utils::accuracy(actualYs, predYs) << std::endl;

	delete root;
}
