//
// Created by André Schnabel on 23.10.15.
//

#include <cmath>
#include <regex>
#include <fstream>
#include <iostream>

#include "Utils.h"
#include <cassert>


using namespace std;

string Utils::slurp(const string &filename) {
	std::ifstream fp(filename);
	if(!fp) throw std::runtime_error("Unable to open file: " + filename);
	string s((std::istreambuf_iterator<char>(fp)), std::istreambuf_iterator<char>());
	return s;
}

vector<string> Utils::readLines(const string &filename) {
    vector<string> lines;
    string line;
	std::ifstream f(filename);
    if(!f) throw std::runtime_error("Unable to open file: " + filename);
    while(!f.eof()) {
		std::getline(f, line);
        lines.push_back(line);
    }
    return lines;
}

int Utils::extractIntFromStr(const string &s, const string &rx) {
	std::regex rxObj(rx);
	std::smatch result;
	std::regex_search(s, result, rxObj);
    return std::stoi(result[1]);
}

vector<int> Utils::extractIntsFromLine(const string &line) {
    const char delim = ' ';
    vector<int> nums;
    string part;
    for(auto c : line) {
        if(c == delim && !part.empty()) {
            nums.push_back(stoi(part));
            part = "";
        } else if(isdigit(c)) {
            part += c;
        }
    }
    if(!part.empty()) {
        nums.push_back(stoi(part));
    }
    return nums;
}

void Utils::serializeSchedule(const vector<int> &sts, const string &filename) {
	std::ofstream f(filename);
	if(f.is_open()) {
		for (int j = 0; j < sts.size(); j++) {
			f << (j + 1) << "->" << sts[j];
			if (j < sts.size() - 1) f << "\n";
		}
		f.close();
	}
	
}

void Utils::serializeProfit(float profit, const string &filename) {
	spit(std::to_string(profit), filename);
}

int Utils::pickWithDistribution(vector<float> &probs, float q) {
	float cumulatedProbs = 0.0f;
	int lastPossibleIx = 0;
	for(int i = 0; i < probs.size(); i++) {
		if (probs[i] > 0.0f && q >= cumulatedProbs && q <= cumulatedProbs + probs[i])
			return i;
		cumulatedProbs += probs[i];
		if(probs[i] > 0.0f) lastPossibleIx = i;
	}
	return lastPossibleIx;
}

void Utils::spit(const string &s, const string &filename) {
	std::ofstream f(filename);
    if(f.is_open()) {
        f << s;
        f.close();
    }
}


void Utils::spitAppend(const string &s, const string &filename) {
    std::ofstream f(filename, std::ios_base::app);
    if(f.is_open()) {
        f << s;
        f.close();
    }
}


string Utils::formattedNow() {
	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%d-%m-%Y %I:%M:%S", timeinfo);
	string str(buffer);
	return str;
}


vector<string> Utils::parseArgumentList(int argc, const char** argv) {
	return Utils::constructVector<string>(argc-1, [&argv](int i) {
		return argv[i+1];
	});
}

bool Utils::fileExists(const std::string &filename) {
	FILE *fp = fopen(filename.c_str(), "r");
	if(fp != nullptr) {
		fclose(fp);
		return true;
	}
	return false;
}

float Utils::average(const std::vector<int> &values) {
	return static_cast<float>(std::accumulate(values.begin(), values.end(), 0)) / static_cast<float>(values.size());
}

float Utils::average(const std::vector<float> &values) {
	return static_cast<float>(std::accumulate(values.begin(), values.end(), 0.0f)) / static_cast<float>(values.size());
}

int Utils::sum(const std::vector<int> &values) {
	int acc = 0;
	for(int i=0; i<values.size(); i++)
		acc += values[i];
	return acc;
}

float Utils::variance(const std::vector<float> &values) {
	if(values.size() <= 1) return 0.0f;
	const float mean = average(values);
	float acc = 0.0f;
	for(int v : values) {
		acc += (v - mean)*(v - mean);
	}
	return acc / static_cast<float>(values.size() - 1);
}

Matrix<char> Utils::transitiveClosure(const Matrix<char> & mx) {
	assert(mx.getM() == mx.getN());
	int n = mx.getN();
	Matrix<char> tc = mx;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
				if (tc(i, k) && tc(k, j)) tc(i, j) = 1;
	return tc;
}

vector<string> Utils::split(const string &s, char sep, char skip_char) {
	vector<string> parts;
	string part;
	for(auto i = s.begin(); i != s.end(); ++i) {
		const char c = *i;

		if (skip_char != 0 && c == skip_char)
			continue;

		if(c == sep) {
			parts.push_back(part);
			part = "";
		} else {
			part.push_back(c);
		}
	}

	if(!part.empty()) {
		parts.push_back(part);
	}

	return parts;
}

Matrix<float> Utils::parseCsvValues(const string& filename) {
	auto lines = readLines(filename);
	vector<vector<float>> data(lines.size()-2);
	int ctr = 0;
	for(const auto &line : lines) {
		if(line.empty()) continue;
		if(ctr > 0) {
			const vector<string> parts = split(line, ',');
			data[ctr-1] = Utils::constructVector<float>(parts.size(), [&parts](int i) { return stof(parts[i]); });
		}
		ctr++;
	}
	return data;
}

std::pair<Matrix<float>, Matrix<float>>
Utils::trainValidationSplit(const Matrix<float> &instances, float trainPercentage, int seed, bool shuffle) {
	std::srand(seed);
	vector<vector<float>> rows = Utils::constructVector<vector<float>>(instances.getM(), [&instances](int i) { return instances.row(i); });
	if(shuffle)
		std::random_shuffle(rows.begin(), rows.end());
	int ninstances = rows.size();
	int divInstanceIndex = (int)round((float)ninstances * trainPercentage);
	int nfeatures = rows[0].size();
	std::pair<Matrix<float>, Matrix<float>> res = make_pair<Matrix<float>, Matrix<float>>(
			Matrix<float>(divInstanceIndex,nfeatures,[&rows](int i, int j) { return rows[i][j]; }),
			Matrix<float>(ninstances-divInstanceIndex,nfeatures,[&rows,divInstanceIndex](int i, int j) { return rows[i+divInstanceIndex][j];})
	);
	return res;
}

float Utils::accuracy(const std::vector<int> &actualYs, const std::vector<int> &predYs) {
	int ncorrect = 0;
	int ntotal = actualYs.size();
	for(int i=0; i<ntotal; i++) {
		if(actualYs[i] == predYs[i]) {
			ncorrect++;
		}
	}
	return (float)ncorrect / (float)ntotal;
}


