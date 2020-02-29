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
	Matrix(int numRows, int numCols, bool isRandom);      //长，宽，是否用随机数初始化，false就用0初始化

	Matrix *transpose();   //转置
	Matrix *copy();      //复制
	 
	void setValue(int r, int c, double v) { this->values.at(r).at(c) = v; }   //给(x,y)像素设值
	double getValue(int r, int c) { return this->values.at(r).at(c); }    //获取(x,y)像素值

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
