#include <bits/stdc++.h>

using namespace std;

auto mnist(const string& images_filepath,const string& labels_filepath) {//////////reader

    vector<uint8_t> labels;{
        ifstream file(labels_filepath, ios::binary);
        uint32_t magic, size;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));file.read(reinterpret_cast<char*>(&size), sizeof(size));

        magic=__builtin_bswap32(magic);
        size=__builtin_bswap32(size);

        labels.resize(size);
        file.read(reinterpret_cast<char*>(labels.data()), size);
    }

    vector<vector<uint8_t>> images;{
        ifstream file(images_filepath, ios::binary);
        uint32_t magic, size, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));file.read(reinterpret_cast<char*>(&size), sizeof(size));file.read(reinterpret_cast<char*>(&rows), sizeof(rows));file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        magic=__builtin_bswap32(magic);
        size=__builtin_bswap32(size);
        rows=__builtin_bswap32(rows);
        cols=__builtin_bswap32(cols);

        vector<uint8_t> image_data(size * rows * cols);
        file.read(reinterpret_cast<char*>(image_data.data()), size * rows * cols);

        images.resize(size, vector<uint8_t>(rows * cols));
        for (size_t i = 0; i < size; ++i) {
            copy(image_data.begin() + i * rows * cols, image_data.begin() + (i + 1) * rows * cols, images[i].begin());
        }
    }

    return make_pair(images, labels);
}

double sigmoid(double temp){//////logistic sigmoid
    double answer = 1 / (1 + 1/exp(temp));
    return answer;
}

int main() {
    /////////////////////data from mnist with mnist reader/////////////////////
    auto trainset = mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");auto testset = mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");auto& train_images = trainset.first;auto& train_labels = trainset.second;auto& test_images = testset.first;auto& test_labels = testset.second;
    //////////////////////////////////////////////////////////////////////////local time 0.155 seconds

    //////////////////////better data container+data copy////////////////////////////////
    vector<vector<double>> trainerimg;
    vector<int> trainerlabels;
    vector<vector<double>> testerimg;
    vector<int> testerlabels;
    for(int j=0;j<60000;j++){
        vector<double> temp;
        for(int i=0;i<784;i++){
            temp.push_back(static_cast<double>(train_images[j][i]));
        }
        trainerimg.push_back(temp);
        trainerlabels.push_back(static_cast<int>(train_labels[j]));
    }
    for(int j=0;j<10000;j++){
        vector<double> temp;
        for(int i=0;i<784;i++){
            temp.push_back(static_cast<double>(test_images[j][i]));
        }
        testerimg.push_back(temp);
        testerlabels.push_back(static_cast<int>(test_labels[j]));
    }
    //////////////////////////////////////////////////////////////////////////local time 1.50 seconds

    ////////////////////normalising pic by pic//////////////////////////////
    for(int j=0;j<60000;j++){
        double temp=0;
        for(int i=0;i<784;i++){
            if(trainerimg[j][i]>temp){
                temp=trainerimg[j][i];
            }
        }
        for(int i=0;i<784;i++){
            trainerimg[j][i]/=temp;
        }
    }
    for(int j=0;j<10000;j++){
        double temp=0;
        for(int i=0;i<784;i++){
            if(testerimg[j][i]>temp){
                temp=testerimg[j][i];
            }
        }
        for(int i=0;i<784;i++){
            testerimg[j][i]/=temp;
        }
    }
    /////////////////////////////////////////////////////////////////////////local time 1.82 seconds

    //////////////////neural network init//////////////////////////////////////
    int inputlayer=784;
    int hlayer1=114;
    int hlayer2=33;
    int outputlayer=10;
    double w1[inputlayer][hlayer1];
    double w2[hlayer1][hlayer2];
    double w3[hlayer2][outputlayer];
    ////////////////////////////////////////////////////////////////////////////local time 2 seconds

    //////////////////////xavier normalised weight init///////////////////
    srand(time(0));
    for(int i=0;i<inputlayer;i++){
        for(int j=0;j<hlayer1;j++){
            double num=(rand()%200-100)*(sqrt(6)/sqrt(inputlayer+hlayer1));
            num/=100;
            w1[i][j]=num;
        }
    }
    for(int i=0;i<hlayer1;i++){
        for(int j=0;j<hlayer2;j++){
            double num=(rand()%200-100)*(sqrt(6)/sqrt(hlayer1+hlayer2));
            num/=100;
            w2[i][j]=num;
        }
    }
    for(int i=0;i<hlayer2;i++){
        for(int j=0;j<outputlayer;j++){
            double num=(rand()%200-100)*(sqrt(6)/sqrt(hlayer2+outputlayer));
            num/=100;
            w3[i][j]=num;
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////local time unchanged

    ////////////////////////////////init of finalchange arrays//////////////////////////////////////////////////////////////
    double finalchangew3[hlayer2][outputlayer];
    double finalchangeb3[outputlayer]={0};
    double finalchangew2[hlayer1][hlayer2];
    double finalchangeb2[hlayer2]={0};
    double finalchangew1[inputlayer][hlayer1];
    double finalchangeb1[hlayer1]={0};
    for(int i=0;i<hlayer2;i++){
        for(int j=0;j<outputlayer;j++){
                finalchangew3[i][j]=0;
        }
    }
    for(int i=0; i<hlayer1; i++){
        for(int j=0; j<hlayer2; j++){
            finalchangew2[i][j]=0;
        }
    }
    for(int i=0; i<inputlayer; i++){
        for(int j=0; j<hlayer1; j++){
            finalchangew1[i][j]=0;
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////local time unchanged

    //////////////////////////////////////forwardprop and backprop(stochastic gradient descent)///////////////////////////////////////////
    int epoch=1;

    double y[10]={0,0,0,0,0,0,0,0,0,0};

    double b1[hlayer1]={0};
    double b2[hlayer2]={0};
    double b3[outputlayer]={0};

    for(int h=0;h<epoch;h++){


    cout<<"\nRunning epoch: "<<h+1<<"\n";
    for(int counter=0;counter<60000;counter++){
    double lnrate=0.0625;//0.125
    double z1[hlayer1]={0};
    double a1[hlayer1];
    double z2[hlayer2]={0};
    double a2[hlayer2];
    double z3[outputlayer]={0};
    double a3[outputlayer];

    double presigmoidz1[hlayer1];
    double presigmoidz2[hlayer2];
    double presigmoidz3[outputlayer];
    double presigmoidz3cost[outputlayer];
        for(int i=0;i<10;i++){
            y[i]=0;
        }
        y[trainerlabels[counter]]=1;
;
        for(int i=0;i<inputlayer;i++){
            if(trainerimg[counter][i]==0){
                continue;
            }
            for(int j=0;j<hlayer1;j++){
                z1[j]+=trainerimg[counter][i]*w1[i][j];
            }
        }

        for(int i=0;i<hlayer1;i++){
            z1[i]+=b1[i];
            a1[i]=sigmoid(z1[i]);
        }

        for(int i=0;i<hlayer1;i++){
            for(int j=0;j<hlayer2;j++){
                z2[j]+=a1[i]*w2[i][j];
            }
        }
        for(int i=0;i<hlayer2;i++){
            z2[i]+=b2[i];
            a2[i]=sigmoid(z2[i]);
        }

        for(int i=0;i<hlayer2;i++){
            for(int j=0;j<outputlayer;j++){
                z3[j]+=a2[i]*w3[i][j];
            }
        }
        for(int i=0;i<outputlayer;i++){
            z3[i]+=b3[i];
            a3[i]=sigmoid(z3[i]);
        }

        double sum=0;
        for(int i=0;i<10;i++){
            sum+=(y[i]-a3[i])*(y[i]-a3[i]);
        }
       // cout<<sum<<" to ";
        for(int i=0;i<hlayer1;i++){
            presigmoidz1[i]=sigmoid(z1[i])*(1-sigmoid(z1[i])*sigmoid(z1[i]));
        }
        for(int i=0;i<hlayer2;i++){
            presigmoidz2[i]=sigmoid(z2[i])*(1-sigmoid(z2[i])*sigmoid(z2[i]));
        }
        for(int i=0;i<outputlayer;i++){
            presigmoidz3[i]=sigmoid(z3[i])*(1-sigmoid(z3[i])*sigmoid(z3[i]));
            presigmoidz3cost[i]=sigmoid(z3[i])*(1-sigmoid(z3[i])*sigmoid(z3[i]))*2*(a3[i]-y[i]);
        }

        for(int i=0;i<hlayer2;i++){
            for(int j=0;j<outputlayer;j++){
                finalchangew3[i][j]+=a2[i]*presigmoidz3[j]*2*(a3[j]-y[j]);
            }
        }
        for(int i=0;i<outputlayer;i++){
            finalchangeb3[i]+=presigmoidz3[i]*2*(a3[i]-y[i]);
        }

        for(int i=0;i<hlayer1;i++){
            for(int j=0;j<hlayer2;j++){
                double sum=0;
                for(int k=0;k<outputlayer;k++){
                    sum+=a1[i]*presigmoidz2[j]*w3[j][k]*presigmoidz3[k]*2*(a3[k]-y[k]);
                }
                finalchangew2[i][j]+=sum;
            }
        }
        for(int j=0; j<hlayer2; j++){
            double sum=0;
            for(int k=0; k<outputlayer; k++)
            {
                sum+=presigmoidz2[j]*w3[j][k]*presigmoidz3[k]*2*(a3[k]-y[k]);
            }
            finalchangeb2[j]+=sum;
        }

        double precalc3[hlayer2];
        for(int k=0;k<hlayer2;k++){
            double sum=0;
            for(int l=0;l<outputlayer;l++){
                sum+=w3[k][l]*presigmoidz3cost[l];
            }
            precalc3[k]=sum;
        }
        for(int i=0;i<inputlayer;i++){
            for(int j=0;j<hlayer1;j++){
                if(trainerimg[counter][i]==0){
                    continue;
                }
                double sum=0;
                double precalc1=trainerimg[counter][i]*presigmoidz1[j];
                for(int k=0;k<hlayer2;k++){
                    double precalc2=w2[j][k]*presigmoidz2[k];
                    sum+=precalc1*precalc2*precalc3[k];
                }
                finalchangew1[i][j]+=sum;
            }
        }
        for(int j=0;j<hlayer1;j++){
            double sum=0;
            for(int k=0;k<hlayer2;k++){
                for(int l=0;l<outputlayer;l++){
                    sum+=presigmoidz1[j]*w2[j][k]*presigmoidz2[k]*w3[k][l]*presigmoidz3cost[l];
                }
            }
            finalchangeb1[j]+=sum;
        }

        double div=1;
        for(int i=0;i<hlayer2;i++){
            for(int j=0;j<outputlayer;j++){
                w3[i][j]-=lnrate*finalchangew3[i][j]/div;
                finalchangew3[i][j]=0;
            }
        }
        for(int j=0;j<outputlayer;j++){
            finalchangeb3[j]/=div;
            b3[j]-=finalchangeb3[j]*lnrate;
            finalchangeb3[j]=0;
        }
        for(int i=0; i<hlayer1; i++){
            for(int j=0; j<hlayer2; j++){
                finalchangew2[i][j]/=div;
                w2[i][j]-=finalchangew2[i][j]*lnrate;
                finalchangew2[i][j]=0;
            }
        }
        for(int j=0; j<hlayer2; j++){
            finalchangeb2[j]/=div;
            b2[j]-=finalchangeb2[j]*lnrate;
            finalchangeb2[j]=0;
        }
        for(int i=0; i<inputlayer; i++){
            for(int j=0; j<hlayer1; j++){
                finalchangew1[i][j]/=div;
                w1[i][j]-=finalchangew1[i][j]*lnrate;
                finalchangew1[i][j]=0;
            }
        }
        for(int j=0; j<hlayer1; j++){
            finalchangeb1[j]/=div;
            b1[j]-=finalchangeb1[j]*lnrate;
            finalchangeb1[j]=0;
        }


        for(int i=0;i<hlayer1;i++){
            z1[i]=0;
        }
        for(int i=0;i<hlayer2;i++){
            z2[i]=0;
        }
        for(int i=0;i<outputlayer;i++){
            z3[i]=0;
        }

        for(int i=0;i<inputlayer;i++){
            if(trainerimg[counter][i]==0){
                continue;
            }
            for(int j=0;j<hlayer1;j++){
                z1[j]+=trainerimg[counter][i]*w1[i][j];
            }
        }

        for(int i=0;i<hlayer1;i++){
            z1[i]+=b1[i];
            a1[i]=sigmoid(z1[i]);
        }

        for(int i=0;i<hlayer1;i++){
            for(int j=0;j<hlayer2;j++){
                z2[j]+=a1[i]*w2[i][j];
            }
        }
        for(int i=0;i<hlayer2;i++){
            z2[i]+=b2[i];
            a2[i]=sigmoid(z2[i]);
        }

        for(int i=0;i<hlayer2;i++){
            for(int j=0;j<outputlayer;j++){
                z3[j]+=a2[i]*w3[i][j];
            }
        }
        for(int i=0;i<outputlayer;i++){
            z3[i]+=b3[i];
            a3[i]=sigmoid(z3[i]);
        }

        sum=0;
        for(int i=0;i<10;i++){
            sum+=(y[i]-a3[i])*(y[i]-a3[i]);
        }
       // cout<<sum<<" \n ";

    }///////////////////440 seconds for local time
    }

    /////////////////////////////test runs///////////////////////

    cout<<"Running tests:\n";
    int currentlabel;
    double correct=0;
    double cost=0;
    int numwise[10]={0};
    int numwisetotal[10]={0};
    for(int counter=0;counter<10000;counter++){
    double z1[hlayer1]={0};
    double a1[hlayer1];
    double z2[hlayer2]={0};
    double a2[hlayer2];
    double z3[outputlayer]={0};
    double a3[outputlayer];
        currentlabel=testerlabels[counter];

        for(int i=0;i<inputlayer;i++){
            for(int j=0;j<hlayer1;j++){
                z1[j]+=testerimg[counter][i]*w1[i][j];
            }
        }
        for(int i=0;i<hlayer1;i++){
            z1[i]+=b1[i];
            a1[i]=sigmoid(z1[i]);
        }

        for(int i=0;i<hlayer1;i++){
            for(int j=0;j<hlayer2;j++){
                z2[j]+=a1[i]*w2[i][j];
            }
        }
        for(int i=0;i<hlayer2;i++){
            z2[i]+=b2[i];
            a2[i]=sigmoid(z2[i]);
        }

        for(int i=0;i<hlayer2;i++){
            for(int j=0;j<outputlayer;j++){
                z3[j]+=a2[i]*w3[i][j];
            }
        }
        double temp=-10000;
        int pos=-1;
        for(int i=0;i<outputlayer;i++){
            z3[i]+=b3[i];
            a3[i]=sigmoid(z3[i]);
            if(a3[i]>temp){
                pos=i;
                temp=a3[i];
            }
        }
        if(pos==currentlabel){
            correct++;
            numwise[pos]++;
        }
        numwisetotal[currentlabel]++;
    }
    double accuracy = correct/100;
    ofstream w("results.txt");
    for(int i=0;i<10;i++){
        w<<numwise[i]<<" / "<<numwisetotal[i]<<" for the number : "<<i<<"\n";
    }
    w<<"Obtained "<<accuracy<<"% accuracy\n";
    cout<<correct/100<<"% accuracy\n";

    return 0;
}
