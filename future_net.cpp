#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <cassert>
using namespace std;
const double A = 30.0;
const double B = 10;
const int layerNum = 3;
const int maxNum = 10;
const double alpha = 0.0035;
const double EPS = 0.002 ;


double sigmoid(double x)
{
    return A / (1 + exp(-x / B));
}



class bp
{
public:
    void calculate(const vector<double> &x);
    void train(const vector <double> &input, const vector <double> &output);
    void setInputNum(const int data)
    {
        inputNum = data;
    }
    void setOutputNum(const int data)
    {
        outputNum = data;
    }
    vector<double> getOutput() const
    {
        vector<double> ret;
        for(auto _data : node[1])
        {
            ret.push_back(_data);
        }
        return ret;
    }
    bp():hideNum(1){
        memset(w,0,sizeof(w));
        memset(b,0,sizeof(b));
    };
    ~bp(){};

private:
    int hideNum;
    int inputNum;
    int outputNum;
    double w[layerNum][maxNum][maxNum]; // Weight of each edge
    double node[layerNum][maxNum];
    double b[layerNum][maxNum];
    double delta[layerNum][maxNum];
    vector<double> output;
    vector<double> input;
private:
    bool isTrainOver();
    void backPropagation();
    // Use the expected output data to train the ANN.



    void setOutput(const vector <double> &data)
    {
        assert((int)data.size() == outputNum);
        output.assign(data.begin(),data.end());
    }
    void setInput(const vector <double> &data)
    {
        assert((int)data.size() == inputNum);
        input.assign(data.begin(),data.end());
    }
    double actFun(double x) const
    {
        return sigmoid(x);
    }
    // The derivative of activation function
    double actFunDeri(double x) const
    {
        return 1.0 * x * (A - x) / (A * B);
    }
};

void bp::backPropagation()
{
    // Calculate the delta value of each node
    for(int i = 0; i < outputNum; i++)
    {
        delta[1][i] = (node[1][i] - output[i]) * actFunDeri(node[1][i]);

    }
    for(int i = 0; i < hideNum; i++)
    {
        double res = 0;
        for(int j = 0; j < outputNum; j++)
        {
            res += delta[1][j] * w[1][i][j];
        }
        delta[0][i] = res * actFunDeri(node[0][i]);
    }

    // Gradient descent procedure

    for(int i = 0; i < inputNum; i++)
    {
        for(int j = 0; j < hideNum; j++)
        {
            w[0][i][j] -= alpha * input[i] * delta[0][j];
        }
    }
    for(int i = 0; i < hideNum; i++)
    {
        for(int j = 0; j < outputNum; j++)
        {
            w[1][i][j] -= alpha * node[0][i] * delta[1][j];
        }
    }
    for(int i = 0; i < outputNum; i++)
    {
        b[1][i] -= alpha * delta[1][i];
    }
    for(int i = 0; i < hideNum; i++)
    {
        b[0][i] -= alpha * delta[0][i];
    }
}


void bp::train(const vector <double> &_input, const vector <double> &_output)
{
    setInput(_input);
    setOutput(_output);
    while(!isTrainOver())
    {
        backPropagation();
    }
    cout << "input : " << w[0][0][0] << "    " << w[0][1][0] << endl;
    cout << "output : " << w[1][0][0] << endl;
}



bool bp::isTrainOver()
{
    bool ret = true;
    calculate(input);
    double res = 0;
    for(int i = 0; i < outputNum; i++)
    {
        res += 0.5 * (output[i] - node[1][i]) * (output[i] - node[1][i]);
        //cout << output[i] << " " << node[1][i] << " "<< res << endl;
    }

    return res < EPS;
}





void bp::calculate(const vector<double> &_data)
{
    //　Calculate each hide node's value
    for(int i = 0; i < hideNum; i++)
    {
        double res = 0;
        for(int j = 0; j < inputNum; j++)
        {
            res += w[0][j][i] * _data[j];
        }
        res += b[0][i];
        node[0][i] = actFun(res);
    }
    // Calculate each output node's value
    for(int i = 0; i < outputNum; i++)
    {
        double res = 0;
        for(int j = 0; j < hideNum; j++)
        {
            res += w[1][j][i] * node[0][j];
        }
        res += b[1][i];
        //cout << res << "   " << sigmoid(res) << endl;
        node[1][i] = sigmoid(res);
    }
}



int main()
{
    bp ann;
    ann.setInputNum(3);
    ann.setOutputNum(1);
    ann.train(vector<double>{-100,-100,-100},vector<double>{1});
    ann.train(vector<double>{100,100,100},vector<double>{0});

    //ann.train(vector<double>{100,100},vector<double>{0});
    while(1)
    {
    double x, y ,z;
    cin >> x >> y >> z;
    ann.calculate(vector<double>{x,y,z});
    vector<double> ans = ann.getOutput();
    cout << ans[0];
    }
    return 0;
}
