#pragma once
#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;

class Matrix
{
public:
	Matrix(int numRows, int numCols, bool isRandom);      //�������Ƿ����������ʼ����false����0��ʼ��

	Matrix *transpose();   //ת��
	Matrix *copy();      //����
	 
	void setValue(int r, int c, double v) { this->values.at(r).at(c) = v; }   //��(x,y)������ֵ
	double getValue(int r, int c) { return this->values.at(r).at(c); }    //��ȡ(x,y)����ֵ

	vector< vector<double> > getValues() { return this->values; }   

	void PrintResult();

	int getNumRows() { return this->numRows; }
	int getNumCols() { return this->numCols; }

private:
	double generateRandomNumber();

	int numRows;
	int numCols;

	vector< vector<double> > values;
};

#endif
