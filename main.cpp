
#include <iostream>
#include "Matrix.h"
#include "Utils.h"
#include "Cart.hpp"

int main() {
	const auto instances = Utils::parseCsvValues("dataforcshc.csv");
	Node *root = buildTree(instances);
	int predY = predictWithTree(root, instances.row(0));
	std::cout << "Predicted class = " << predY << std::endl;
	delete root;
}
