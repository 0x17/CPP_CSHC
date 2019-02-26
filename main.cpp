
#include <iostream>
#include <cmath>
#include "Matrix.h"
#include "Utils.h"
#include "Cart.hpp"
#include "Bagging.hpp"

using namespace std;

template <typename T> string tostr(const T& t) {
	ostringstream os;
	os<<t;
	return os.str();
}

template<class Func>
float accuracyOnSamples(Func predictionFunc, const Matrix<float> &samples) {
	int classIndex = samples.getN()-1;
	vector<int> actualYs = Utils::constructVector<int>(samples.getM(), [classIndex,&samples](int i) { return samples(i,classIndex); });
	vector<int> predYs = predictionFunc(samples);
	return Utils::accuracy(actualYs, predYs);
}

void serializeMatrixToCsv(const Matrix<float> &mx, const std::string &outFilename, bool withHeader = true) {
	std::string ostr;

	if(withHeader) {
		for(int j=0; j<mx.getN(); j++) {
			ostr += "attr"+to_string(j) + ((j+1) < mx.getN() ? "," : "");
		}
		ostr += '\n';
	}

	for(int i=0; i<mx.getM(); i++) {
		for(int j=0; j<mx.getN(); j++) {
			ostr += tostr(mx(i,j)) + ((j + 1) < mx.getN() ? "," : "");
		}
		ostr += '\n';
	}
	Utils::spit(ostr, outFilename);
}

int main() {
	const bool useEnsemble = false;

	const auto instances = Utils::parseCsvValues("dataforcshc.csv");

	const auto res = Utils::trainValidationSplit(instances, 0.9, 23, true);
	const auto train = res.first;
	const auto validation = res.second;

	cout << "Train size = " << train.getM() << ", validation size = " << validation.getM() << endl;

	serializeMatrixToCsv(train, "mytrain.csv");

	function<vector<int>(const Matrix<float> &)> predFunc;
	function<void(void)> freeFunc;

	if(useEnsemble) {
		//const int numFeatures = static_cast<const int>(round(sqrt(train.getN() - 1.0f) * 0.7f));
		//const int subSampleSize = static_cast<const int>(round(0.7f * train.getM()));
		//const int numTrees = 500;

		const int numFeatures = 80;
		const int subSampleSize = 1800;
		const int numTrees = 8;

		const auto ensemble = createEnsemble(numTrees, instances, numFeatures, subSampleSize);
		predFunc = [&ensemble](const Matrix<float> &samples) { return predictWithEnsemble(ensemble, samples); };

		freeFunc = [&ensemble]() {
			for(Node *root : ensemble) {
				delete root;
			}
		};
	} else {
		Node *root = buildTree(train);
		predFunc = [&root](const Matrix<float> &samples) { return predictWithTree(root, samples); };
		freeFunc = [&root]() {
			delete root;
		};
	}

	float accTrain = accuracyOnSamples(predFunc, train);
	float accVal = accuracyOnSamples(predFunc, validation);

	freeFunc();

	cout << "Training accuracy = " << accTrain << endl;
	cout << "Prediction accuracy = " << accVal << endl;
}
