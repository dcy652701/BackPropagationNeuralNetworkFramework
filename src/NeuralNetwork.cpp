#include"include/NeuralNetwork.hpp"
#include"include/utils/Matrix_Math.hpp"

NeuralNetwork::NeuralNetwork(ANNConfig config) {
	this->config = config;
	this->NeuralNetworkStructure = config.NeuralNetworkStructure;
	this->NeuralNetworkStructureSize = config.NeuralNetworkStructure.size();
	this->learningRate = config.learningRate;
	this->momentum = config.momentum;
	this->bias = config.bias;

	this->hiddenActivationType = config.hActivation;
	this->outputActivationType = config.oActivation;
	this->costFunctionType = config.cost;

	for (int i = 0; i < NeuralNetworkStructureSize; i++) {
		if (i > 0 && i < (NeuralNetworkStructureSize - 1)) {
			Layer *l = new Layer(NeuralNetworkStructure.at(i), this->hiddenActivationType);
			this->layers.push_back(l);
		}
		else if (i == (NeuralNetworkStructureSize - 1)) {
			Layer *l = new Layer(NeuralNetworkStructure.at(i), this->outputActivationType);
			this->layers.push_back(l);
		}
		else {
			Layer *l = new Layer(NeuralNetworkStructure.at(i));
			this->layers.push_back(l);
		}
	}

	for (int i = 0; i < (NeuralNetworkStructureSize - 1); i++) {
		Matrix *m = new Matrix(NeuralNetworkStructure.at(i), NeuralNetworkStructure.at(i + 1), true);

		this->weightMatrices.push_back(m);
	}

	// Initialize empty errors
	for (int i = 0; i < NeuralNetworkStructure.at(NeuralNetworkStructure.size() - 1); i++) {
		errors.push_back(0.00);
		derivedErrors.push_back(0.00);
	}

	this->error = 0.00;
}

void NeuralNetwork::train(vector<double> input, vector<double> target, double bias, double learningRate, double momentum)
{
	this->learningRate = learningRate;     //设置学习率
	this->momentum = momentum; //设置动量
	this->bias = bias;  //设置偏置

	this->setCurrentInput(input);   //设置当前的输入  
	this->setCurrentTarget(target);//设置输出

	this->feedForward();//向前传播
	this->lostFunction();//计算误差
	this->backPropagation();//反向传播
}

void NeuralNetwork::setCurrentInput(vector<double> input)
{
	this->input = input;

	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i));
	}
}

void NeuralNetwork::feedForward()
{
	Matrix *a;  //第一层
	Matrix *b;  //隐藏层
	Matrix *c;  //输出层

	for (int i = 0; i < (this->NeuralNetworkStructureSize - 1); i++) 
	{
		a = this->getNeuronMatrix(i);
		b = this->getWeightMatrix(i);
		c = new Matrix(a->getNumRows(), b->getNumCols(), false);

		if (i != 0) 
		{
			a = this->getActivatedNeuronMatrix(i);
		}

		::utils::Math::multiplyMatrix(a, b, c);

		for (int c_index = 0; c_index < c->getNumCols(); c_index++) 
		{
			this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index) + this->bias);
		}

		delete a;
		delete b;
		delete c;
	}
}


void NeuralNetwork::backPropagation()
{
	vector<Matrix *> newWeights;
	Matrix *deltaWeights;
	Matrix *gradients;
	Matrix *derivedValues;
	Matrix *gradientsTransposed;
	Matrix *zActivatedVals;
	Matrix *tempNewWeights;
	Matrix *pGradients;
	Matrix *transposedPWeights;
	Matrix *hiddenDerived;
	Matrix *transposedHidden;

	//从输出层到隐藏层
	int indexOutputLayer = this->NeuralNetworkStructure.size() - 1;

	gradients = new Matrix(1, this->NeuralNetworkStructure.at(indexOutputLayer), false);

	derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();

	for (int i = 0; i < this->NeuralNetworkStructure.at(indexOutputLayer); i++) {
		double e = this->derivedErrors.at(i);
		double y = derivedValues->getValue(0, i);
		double g = e * y;
		gradients->setValue(0, i, g);
	}
	//gradient*z
	gradientsTransposed = gradients->transpose();
	zActivatedVals = this->layers.at(indexOutputLayer - 1)->matrixifyActivatedVals();

	deltaWeights = new Matrix(
		gradientsTransposed->getNumRows(),
		zActivatedVals->getNumCols(),
		false
	);

	::utils::Math::multiplyMatrix(gradientsTransposed, zActivatedVals, deltaWeights);
	
	//计算输出层与隐藏层之间的新权重
	tempNewWeights = new Matrix(
		this->NeuralNetworkStructure.at(indexOutputLayer - 1),
		this->NeuralNetworkStructure.at(indexOutputLayer),
		false
	);

	for (int r = 0; r < this->NeuralNetworkStructure.at(indexOutputLayer - 1); r++) {
		for (int c = 0; c < this->NeuralNetworkStructure.at(indexOutputLayer); c++) {

			double originalValue = this->weightMatrices.at(indexOutputLayer - 1)->getValue(r, c);
			double deltaValue = deltaWeights->getValue(c, r);

			originalValue = this->momentum * originalValue;
			deltaValue = this->learningRate * deltaValue;

			tempNewWeights->setValue(r, c, (originalValue - deltaValue));
		}
	}

	newWeights.push_back(new Matrix(*tempNewWeights));

	delete gradientsTransposed;
	delete zActivatedVals;
	delete tempNewWeights;
	delete deltaWeights;
	delete derivedValues;


	//隐藏层到输入层
	for (int i = (indexOutputLayer - 1); i > 0; i--) {
		pGradients = new Matrix(*gradients);
		delete gradients;

		transposedPWeights = this->weightMatrices.at(i)->transpose();

		gradients = new Matrix(
			pGradients->getNumRows(),
			transposedPWeights->getNumCols(),
			false
		);

		::utils::Math::multiplyMatrix(pGradients, transposedPWeights, gradients);

		hiddenDerived = this->layers.at(i)->matrixifyDerivedVals();

		for (int colCounter = 0; colCounter < hiddenDerived->getNumRows(); colCounter++) {
			double  g = gradients->getValue(0, colCounter) * hiddenDerived->getValue(0, colCounter);
			gradients->setValue(0, colCounter, g);
		}

		if (i == 1) {
			zActivatedVals = this->layers.at(0)->matrixifyVals();
		}
		else {
			zActivatedVals = this->layers.at(0)->matrixifyActivatedVals();
		}

		transposedHidden = zActivatedVals->transpose();

		deltaWeights = new Matrix(
			transposedHidden->getNumRows(),
			gradients->getNumCols(),
			false
		);

		::utils::Math::multiplyMatrix(transposedHidden, gradients, deltaWeights);

		//更新权重
		tempNewWeights = new Matrix(
			this->weightMatrices.at(i - 1)->getNumRows(),
			this->weightMatrices.at(i - 1)->getNumCols(),
			false
		);

		for (int r = 0; r < tempNewWeights->getNumRows(); r++) {
			for (int c = 0; c < tempNewWeights->getNumCols(); c++) {
				double originalValue = this->weightMatrices.at(i - 1)->getValue(r, c);
				double deltaValue = deltaWeights->getValue(r, c);

				originalValue = this->momentum * originalValue;
				deltaValue = this->learningRate * deltaValue;

				tempNewWeights->setValue(r, c, (originalValue - deltaValue));
			}
		}

		newWeights.push_back(new Matrix(*tempNewWeights));

		delete pGradients;
		delete transposedPWeights;
		delete hiddenDerived;
		delete zActivatedVals;
		delete transposedHidden;
		delete tempNewWeights;
		delete deltaWeights;
	}

	for (int i = 0; i < this->weightMatrices.size(); i++) {
		delete this->weightMatrices[i];
	}

	this->weightMatrices.clear();

	reverse(newWeights.begin(), newWeights.end());

	for (int i = 0; i < newWeights.size(); i++) {
		this->weightMatrices.push_back(new Matrix(*newWeights[i]));
		delete newWeights[i];
	}

}

void NeuralNetwork::MSE_LostFunction()
{
	int outputLayerIndex = this->layers.size() - 1;
	vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();

	this->error = 0.00;

	for (int i = 0; i < target.size(); i++) {
		double t = target.at(i);
		double y = outputNeurons.at(i)->getActivatedVal();

		errors.at(i) = 0.5 * pow(abs((t - y)), 2);    
		derivedErrors.at(i) = (y - t);

		this->error += errors.at(i);
	}
}

void NeuralNetwork::lostFunction()
{
	switch (costFunctionType) 
	{
		case(COST_MSE): 
			this->MSE_LostFunction(); break;
		default: 
			this->MSE_LostFunction(); break;
	}
}

void NeuralNetwork::saveWeights(string file)
{
	json j = {};

	vector< vector< vector<double> > > weightSet;

	for (int i = 0; i < this->weightMatrices.size(); i++)
	{
		weightSet.push_back(this->weightMatrices.at(i)->getValues());
	}

	j["weights"] = weightSet;
	j["NeuralNetworkStructure"] = this->NeuralNetworkStructure;
	j["learningRate"] = this->learningRate;
	j["momentum"] = this->momentum;
	j["bias"] = this->bias;

	std::ofstream o(file);
	o << std::setw(4) << j << endl;
}

void NeuralNetwork::loadWeights(string file)
{
	std::ifstream i(file);
	json jWeights;
	i >> jWeights;

	vector< vector< vector<double> > > temp = jWeights["weights"];

	for (int i = 0; i < this->weightMatrices.size(); i++) 
	{
		for (int r = 0; r < this->weightMatrices.at(i)->getNumRows(); r++) 
		{
			for (int c = 0; c < this->weightMatrices.at(i)->getNumCols(); c++) 
			{
				this->weightMatrices.at(i)->setValue(r, c, temp.at(i).at(r).at(c));
			}
		}
	}
}