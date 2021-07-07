/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        June 2021
--------------------------------------------------
    A simple FFN to digest hexadecimal sha256 hashes.

    This is a POC for NPOW; https://bit.ly/3wMj6ha

    This is a simple feed-forward network where the
    error gradient from each unit of the final layer
    is averaged into one gradient which is then back
    propagated through the prior layers.
    
    The use of hexadecimal is to help the network reach
    a minima using simplified, normalised, character
    embeddings. It is also convenient that sha256 hashes
    are typically exchanged in hexadecimal format.

    Focus on this networking being simple and fast, as
    such only unit dropout and SGD is used with batching.

    Elliot sigmoid / Softsign was chosen as a personal
    favorite; simple, fast, and often with good results.

    It is an intention that weights once trained are packed
    into int8 representations for network transport and then
    unpacked back into float32. For this reason the fromEmbed()
    function was created to error-correct the output due to any
    loss in precision occurring from this conversion.

    ----

    The backprop in this code has bismal optimisation, please
    refer to the python backprop for adequate weight training.

    ----

    Compile:
    gcc -std=gnu99 -Ofast poc0.c -lm -o npow

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define uint unsigned int
#define int8 signed char

///

// variables 
int8  export_weight[64][4160]; // weights to export over network
float weight[64][4160];        // weights calculated in backprop / training
float output[64][64];          // outputs from each layer
float grad[64][64];            // backprop'd gradients for each layer

///

// hyperparameters
uint  layers = 1;         // 
uint  batches = 3;        // !!! NOT IMPLEMENTED !!!
float lrate = 0.0000006;  // 
float dropout = 0.3;      // 
float weight_init = 0.22; // uniform lecun constants

// switches
uint linear_final_layer = 0;     // 0 = final layer has elliot sigmoid; 1 = linear
uint backprop_average_error = 0; // 0 = each final layer unit is corrected by its own error gradient
                                 // 1 = each final layer unit is corrected by an average of all unit gradients

#define last_layer layers-1

///

char in1[] =  "5426ca398cdf52e5acdcc5b7f07f69ab16ccee91c361e649b4d91399048d87c0";
char out1[] = "976d37abad4f34c2be446ea09a1b237246c9755c8dc87e461d52a995bf6ff98d";

// char in2[] =  "b5a6461ef56594a1391e7596ad28a1a188fba5aa4156d1d9ac0c6af2ff5ababa";
// char out2[] = "b184825a70307ee675a71fec8a62acf54ac658371ceb48d6028d7071ddbaa5b0";

// char in3[] =  "beea7fcd54cea2b12816744d09d82274413aa826fc46c6fb479209c6dcded0f6";
// char out3[] = "2b3c84860e74fa728537f5940a08c08dc8110d9d87600faff8a556f90bd4fd65";

time_t lt = 0;
uint ltt = 0;
float le = 0;
float terror = 0;

///

#define SCALE_FACTOR 28  // Recommended factor: 32 - 128
                         // 8   = ~0.13 to 16.0
                         // 16  = ~0.07 to 8.0
                         // 32  = ~0.04 to 4.0
                         // 64  = ~0.02 to 2.0
                         // 128 = ~0.01 to 1.0
int8 packFloat(const float f)
{
    float r = (f * SCALE_FACTOR);
    if(r > 0){r += 0.5;}
    else if(r < 0){r -= 0.5;}
    if(r >= 127){return 127;}
    else if(r <= -128){return -128;}
    return (int8)r;
}
float unpackFloat(const int8 f)
{
    return (float)(f / SCALE_FACTOR);
}

void exportWeights()
{
    for(int i = 0; i < layers; i++)
        for(int j = 0; j < 4160; j++)
            export_weight[i][j] = packFloat(weight[i][j]);
}

///

// exact
float toEmbed(const signed char ic)
{
    signed char c = toupper(ic);
    // returns normalised embedding -0.8828125 to +0.875 range
         if(c == '0'){return -0.8828125;}
    else if(c == '1'){return -0.765625;}
    else if(c == '2'){return -0.6484375;}
    else if(c == '3'){return -0.53125;}
    else if(c == '4'){return -0.4140625;}
    else if(c == '5'){return -0.296875;}
    else if(c == '6'){return -0.1796875;}
    else if(c == '7'){return -0.0625;}
    else if(c == '8'){return  0.0546875;}
    else if(c == '9'){return  0.171875;}
    else if(c == 'A'){return  0.2890625;}
    else if(c == 'B'){return  0.40625;}
    else if(c == 'C'){return  0.5234375;}
    else if(c == 'D'){return  0.640625;}
    else if(c == 'E'){return  0.7578125;}
    else if(c == 'F'){return  0.875;}
}

// tolerance error correcting
char fromEmbed(const float f)
{
    // returns characters from a -1 to +1 range
         if(f > -1 && f < -0.82421875)          {return '0';}
    else if(f >= -0.82421875 && f < -0.70703125){return '1';}
    else if(f >= -0.70703125 && f < -0.58984375){return '2';}
    else if(f >= -0.58984375 && f < -0.47265625){return '3';}
    else if(f >= -0.47265625 && f < -0.35546875){return '4';}
    else if(f >= -0.35546875 && f < -0.23828125){return '5';}
    else if(f >= -0.23828125 && f < -0.12109375){return '6';}
    else if(f >= -0.12109375 && f < -0.00390625){return '7';}
    else if(f >= -0.00390625 && f <  0.11328125){return '8';}
    else if(f >=  0.11328125 && f <  0.23046875){return '9';}
    else if(f >=  0.23046875 && f <  0.34765625){return 'A';}
    else if(f >=  0.34765625 && f <  0.46484375){return 'B';}
    else if(f >=  0.46484375 && f <  0.58203125){return 'C';}
    else if(f >=  0.58203125 && f <  0.69921875){return 'D';}
    else if(f >=  0.69921875 && f <  0.81640625){return 'E';}
    else if(f >=  0.81640625 && f <  1)         {return 'F';}
}

///

float qRandFloat(const float min, const float max)
{
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
}

float qRandWeight(const float min, const float max)
{
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

uint qRand(const uint min, const uint max)
{
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * ((max+1)-min) ) + min;
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srand(time(0)+c.tv_nsec);
}

static inline float elliot_sigmoid(const float x) // aka softsign
{
    return x / (1 + fabs(x));
}

static inline float elliot_sigmoidDerivative(const float x)
{
    const float a = 1 - fabs(x);
    return a*a;
}

///

void rndWeights()
{
    for(int i = 0; i < layers; i++)
    {
        int k = 0;
        for(int j = 0; j < 4160; j++)
        {
            if(k == 65)
            {
                weight[i][j] = 0; // 0 init biases
                k = 0;
            }
            else
            {
                weight[i][j] = qRandWeight(-weight_init, weight_init);
            }
            k++;
        }
    }
}

void rndHyperparameters()
{
    lrate   = qRandFloat(0.001, 0.1);
    dropout = qRandFloat(0, 0.3);
    batches = qRand(1, 3);
    layers  = qRand(2, 64);
    weight_init = qRandFloat(0.1, 0.9);
}

void processNetwork(const float* inputs, const float* targets)
{
    // set initial input
    const float* ni = inputs;

    // forward prop
    for(int i = 0; i < layers; i++)
    {
        int k = 0, u = 0;
        float uo = 0; // unit output
        for(int j = 0; j < 4160; j++)
        {
            if(k == 64)
            {
                uo += weight[i][j]; // bias
                if(linear_final_layer == 1 && u == last_layer)
                    output[i][u] = uo; // linear output on final layer
                else
                    output[i][u] = elliot_sigmoid(uo);
                uo = 0;
                k = 0;
                u++;
            }
            else
            {
                uo += ni[k] * weight[i][j];
                k++;
            }
        }
        ni = &output[i][0];
    }

    // back prop
    if(targets != NULL)
    {
        if(backprop_average_error == 1)
        {
            // compute total error gradient of the output/last layer
            terror = 0;
            for(int i = 0; i < 64; i++)
            {
                if(targets[i] < output[last_layer][i])
                    terror += targets[i] - output[last_layer][i];
                else
                    terror += output[last_layer][i] - targets[i];
            }
            terror /= 64;
            
            // backprop last layer gradients
            for(int i = 0; i < 64; i++)
                grad[last_layer][i] = elliot_sigmoidDerivative(output[last_layer][i]) * terror;
        }
        else
        {
            // compute total error gradient of the output/last layer
            terror = 0;
            
            // backprop last layer gradients
            for(int i = 0; i < 64; i++)
            {
                float err = 0;
                if(targets[i] < output[last_layer][i])
                    err = targets[i] - output[last_layer][i];
                else
                    err = output[last_layer][i] - targets[i];
                terror += err;

                grad[last_layer][i] = elliot_sigmoidDerivative(output[last_layer][i]) * err;
            }
        }

        // write out error every second
        if(time(0) > lt)
        {
            printf("%f ", terror);
            ltt = 1;
        }

        // backprop prior to last layer gradients
        for(int i = last_layer-1; i >= 0; i--)
        {
            // accumulate total error for the layer
            int k = 0, u = 0;
            float ler = 0;

            for(int j = 0; j < 4160; j++)
            {
                ler += weight[i][j] * grad[i+1][u];

                if(k == 64)
                    k = 0, u++;
                else
                    k++;
            }

            // prepare gradients for the prior layer
            for(int j = 0; j < 64; j++)
            {
                grad[i][j] = elliot_sigmoidDerivative(output[i][j]) * ler;
            }
        }

        // forward prop the input layer
        int k = 0, u = 0, j = 0;
        if(dropout != 0 && qRandFloat(0, 1) <= dropout) // dropout
            j = 65, u = 1;
        for(; j < 4160; j++)
        {
            if(k == 64)
            {
                weight[0][j] += lrate * grad[0][u]; // bias

                if(dropout != 0 && qRandFloat(0, 1) <= dropout) // dropout
                        j += 65, u++;
                
                k = 0, u++;
            }
            else
            {
                weight[0][j] += lrate * grad[0][u] * inputs[u];
                k++;
            }
        }

        // forward prop the latent/hidden layers
        for(int i = 1; i < layers; i++)
        {
            int k = 0, u = 0, j = 0;
            if(dropout != 0 && qRandFloat(0, 1) <= dropout) // dropout
                j = 65, u = 1;
            for(; j < 4160; j++)
            {
                if(k == 64)
                {
                    weight[i][j] += lrate * grad[i][u]; // bias

                    if(dropout != 0 && qRandFloat(0, 1) <= dropout) // dropout
                        j += 65, u++;
                    
                    k = 0, u++;
                }
                else
                {
                    weight[i][j] += lrate * grad[i][u] * output[i-1][u];
                    k++;
                }
            }
        }
    }
    // done
}

int main()
{
    // convert input and target to embeddings
    float input[64];
    for(int i = 0; i < 64; i++)
        input[i] = toEmbed(in1[i]);

    float target[64];
    for(int i = 0; i < 64; i++)
        target[i] = toEmbed(out1[i]);

    // program loop
    while(1)
    {
        newSRAND();
        //rndHyperparameters();
        rndWeights();

        // print init weights
        // for(int i = 0; i < layers; i++)
        // {
        //     for(int j = 0; j < 4160; j++)
        //     {
        //         printf("%.2f ", weight[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("------------------------\n\n");

        // iterate indefinitely 
        char out[65];
        uint iteration = 0;
        while(1)
        {
            processNetwork(&input[0], &target[0]);

            for(int i = 0; i < 64; i++)
                out[i] = fromEmbed(output[last_layer][i]);
            out[64] = 0x00;

            if(ltt == 1)
            {
                printf("%i: %s\n", iteration, out);
                // if(floor(terror) == floor(le))
                // {
                //     printf("--------\n");
                //     break;
                // }
                lt = time(0)+1;
                ltt = 0;
                le = terror;
            }

            //sleep(1);
            iteration++;
        }

    }

    return 0;
}

