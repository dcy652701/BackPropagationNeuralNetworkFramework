#include"include/utils/Fetch_Dataset.hpp"

vector< vector<double> > utils::FetchData::fetchData(string path) {
	vector< vector<double> > data;

	ifstream infile(path);

	string line;
	while (getline(infile, line)) {
		vector<double>  dRow;
		string          tok;
		stringstream    ss(line);

		while (getline(ss, tok, ',')) {
			dRow.push_back(stof(tok));
		}

		data.push_back(dRow);
	}

	return data;
}