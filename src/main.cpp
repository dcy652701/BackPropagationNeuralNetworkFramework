#include<iostream>
#include<streambuf>
#include<fstream>
#include"include/Matrix.hpp"
#include"include/json.hpp"
#include"include/NeuralNetwork.hpp"
#include"include/utils/Fetch_Dataset.hpp"


using namespace std;
using json = nlohmann::json;
void print_error()
{
	cout << "no input file" << endl;
}

ANNConfig AnalyzeConfig(json configObject)      //½âÎöconfig.json
{
	ANNConfig config;

	double learningRate = configObject["learningRate"];
	double momentum = configObject["momentum"];
	double bias = configObject["bias"];
	int epoch = configObject["epoch"];
	string trainingFile = configObject["trainingFile"];
	string labelsFile = configObject["labelsFile"];
	string weightsFile = configObject["weightsFile"];
	vector<int> NeuralNetworkStructure = configObject["NeuralNetworkStructure"];

	ANN_ACTIVATION hidden_layer_Activation = configObject["hActivation"];
	ANN_ACTIVATION output_layer_Activation = configObject["oActivation"];

	config.NeuralNetworkStructure = NeuralNetworkStructure;
	config.bias = bias;
	config.learningRate = learningRate;
	config.momentum = momentum;
	config.epoch = epoch;
	config.hActivation = hidden_layer_Activation;
	config.oActivation = output_layer_Activation;
	config.trainingFile = trainingFile;
	config.labelsFile = labelsFile;
	config.weightsFile = weightsFile;

	return config;
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		print_error();
		exit(-1);
	}
	ifstream configFile(argv[1]);
	string str((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
	cout << str << endl;
	//auto res = buildConfig(json::parse(str));
	NeuralNetwork *n = new NeuralNetwork(AnalyzeConfig(json::parse(str)));
	vector< vector<double> > trainingData = utils::FetchData::fetchData(n->config.trainingFile);
	vector< vector<double> > labelData = utils::FetchData::fetchData(n->config.labelsFile);
	

	for (int i = 0; i < n->config.epoch; i++) 
	{
		for (int tIndex = 0; tIndex < trainingData.size(); tIndex++) 
		{
			vector<double> input = trainingData.at(tIndex);
			vector<double> target = labelData.at(tIndex);

			n->train(
				input,
				target,
				n->config.bias,
				n->config.learningRate,
				n->config.momentum
			);
		}
		cout << n->error << endl;

		//cout << "Error at epoch " << i+1 << ": " << n->error << endl;
	}

	//n->saveWeights("D:/NeuralNetworkStructure/weights.json");
	cin.get();
	return 0;
}
