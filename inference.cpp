
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <float.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cblas.h>
#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <fstream>  // NOLINT(readability/streams)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <regex>


#ifdef _DEBUG
#pragma comment(lib, "libprotobufd.lib")
#else
#pragma comment(lib, "libprotobuf.lib")
#endif


#pragma comment(lib, "libopenblas.dll")

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using namespace cv;

int shapeDim(const BlobProto& blob, int axis){

	const auto& shape = blob.shape();
	if (axis < shape.dim_size())
		return shape.dim(axis);
	return 1;
}



void loadmodel(int phase, vector<vector<float>>& weights_conv,
	vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
	string path = "";
	if (phase == 0)
		path = "../models/det1.caffemodel";
	else if (phase == 1)
		path = "../models/det2.caffemodel";
	else
		path = "../models/det3.caffemodel";
	const char* infile = path.c_str();
	int fd = _open(infile, O_RDONLY | O_BINARY);
	if (fd == -1){
		printf("open file[%s] fail.\n", infile);
		return;
	}

	std::shared_ptr<ZeroCopyInputStream> raw_input = std::make_shared<FileInputStream>(fd);
	NetParameter net;
	bool success = net.ParseFromZeroCopyStream(raw_input.get());
	raw_input.reset();
	_close(fd);

	for (int i = 0; i < net.layer_size(); ++i){
		const LayerParameter& layer = net.layer(i);

		printf("layer: %s\n", layer.name().c_str());
		for (int j = 0; j < layer.bottom_size(); ++j)
			printf("bottom_%d: %s\n", j, layer.bottom(j).c_str());

		for (int j = 0; j < layer.top_size(); ++j)
			printf("top_%d: %s\n", j, layer.top(j).c_str());

		for (int j = 0; j < layer.blobs_size(); ++j){
			const BlobProto& params = layer.blobs(j);
			printf("param %d[%dx%dx%dx%d]: ", j, shapeDim(params, 0), shapeDim(params, 1), shapeDim(params, 2), shapeDim(params, 3));

			int num = params.data_size();
			//num = min(10, num);
			vector<float> weights_tmp;
			for (int k = 0; k < num; ++k){
				weights_tmp.emplace_back(params.data(k));
				//printf("%.2f,", params.data(k));
			}
			if (regex_match(layer.name().c_str(), regex("(conv)(.*)", regex::icase)) && j == 0 && ((phase == 1 && i >= 11) || (phase == 2 && i >= 14))){
				weights_fc.emplace_back(weights_tmp);
				continue;
			}
			if (regex_match(layer.name().c_str(), regex("(conv)(.*)", regex::icase)) && j == 1 && ((phase == 1 && i >= 11) || (phase == 2 && i >= 14))){
				bias_fc.emplace_back(weights_tmp);
				continue;
			}
			if (regex_match(layer.name().c_str(), regex("(conv)(.*)", regex::icase)) && j == 0){
				weights_conv.emplace_back(weights_tmp);
				continue;
			}
			if (regex_match(layer.name().c_str(), regex("(conv)(.*)", regex::icase)) && j == 1){
				bias.emplace_back(weights_tmp);
				continue;
			}
			if (regex_match(layer.name().c_str(), regex("(PReLU)(.*)", regex::icase))){
				weights_pr.emplace_back(weights_tmp);
				continue;
			}

		}
	}
}


void storemodel(vector<vector<vector<float>>>& weights_conv, vector<vector<vector<float>>>& bias, vector<vector<vector<float>>>& weights_pr, vector<vector<vector<float>>>& weights_fc, vector<vector<vector<float>>>& bias_fc){

	for (int i = 0; i < 3; ++i){
		vector<vector<float>> weights_conv_tmp;
		vector<vector<float>> bias_tmp;
		vector<vector<float>> weights_pr_tmp;
		vector<vector<float>> weights_fc_tmp;
		vector<vector<float>> bias_fc_tmp;
		loadmodel(i, weights_conv_tmp, bias_tmp, weights_pr_tmp, weights_fc_tmp, bias_fc_tmp);
		weights_conv.emplace_back(weights_conv_tmp);
		bias.emplace_back(bias_tmp);
		weights_pr.emplace_back(weights_pr_tmp);
		weights_fc.emplace_back(weights_fc_tmp);
		bias_fc.emplace_back(bias_fc_tmp);
	}
}


//默认Mat是float类型的,Mat转vector
void Mat2vec(Mat& src, vector<float>& dst){
	//按行展开的
// 	if (src.isContinuous()) {
// 		vector<float> vec;
// 		vec = src.reshape(1, 1).clone();
// 		dst.insert(dst.end(), vec.begin(), vec.end());
// 	}
// 	else{
// 		for (int i = 0; i < src.rows; ++i){
// 			dst.insert(dst.end(), src.ptr<float>(i), src.ptr<float>(i)+src.cols);
// 		}
// 	}
	vector<float> vectmp;
	//vectmp.clear();
	vectmp.assign((float*)src.datastart, (float*)src.dataend);
	dst.insert(dst.end(), vectmp.begin(), vectmp.end());
	//return vec;
}

vector<Mat> Vec2vecmat(vector<float>& vec, int channels, int rows, int cols){
	vector<Mat> dst;
	for (int i = 0; i < channels; ++i){
		vector<float> tmp;
		for (int j = i*rows*cols; j < (i + 1)*rows*cols; ++j){
			tmp.emplace_back(vec[j]);
		}
		Mat dst_tmp(tmp);
		dst_tmp = dst_tmp.reshape(1, rows).clone();
		dst.emplace_back(dst_tmp);
	}
	return dst;
}


//二维数组转化为一维数组
template <typename T>
vector<T> vec2d2vec(vector<vector<T>>& vec2d){
	vector<T> vec;
	for (int i = 0; i < vec2d.size(); ++i){
		for (j = 0; j < vec2d[0].size(); ++j){
			vec[j + i*vec2d.size()] = vec2d[i][j];
		}
	}
}


// vector<float> im2col(int kernel_size, int stride, const vector<Mat>& src, int output_h, int output_w){
// 	vector<float> vec_tmp;
// 	for (int i = 0; i <= src[0].rows - ((src[0].rows - kernel_size) % stride) - kernel_size; i = i + stride){
// 		for (int j = 0; j <= src[0].cols - ((src[0].cols - kernel_size) % stride) - kernel_size; j = j + stride){
// 			for (Mat mat : src){
// 				Rect rect(j, i, kernel_size, kernel_size);
// 				Mat region = mat(rect);
// 				Mat2vec(region, vec_tmp);
// 				//dst.insert(dst.end(), tmp.begin(), tmp.end());
// 			}
// 		}
// 	}
// 	Mat mat = Mat(vec_tmp);//将vector变成单列的mat
// 	Mat dest = mat.reshape(1, output_h*output_w).clone();
// 	mat = dest.t();
// 	vector<float> dst;
// 	Mat2vec(mat, dst);
// 	return dst;
// }

bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


 void im2col(const float* data, const int channels, const int height, const int width, const int kernel, const int pad, const int stride, float* col_buff) {
 	const int dilation = 1;
 	const int output_h = (height + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
 	const int output_w = (width + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
 	const int channel_size = height * width;
 	int count = 0;
 	for (int channel = channels; channel--; data += channel_size) {
 		for (int kernel_row = 0; kernel_row < kernel; kernel_row++) {
 			for (int kernel_col = 0; kernel_col < kernel; kernel_col++) {
 				int input_row = -pad + kernel_row * dilation;
 				for (int output_rows = output_h; output_rows; output_rows--) {
 					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
 						for (int output_cols = output_w; output_cols; output_cols--) {
 							*(col_buff++) = 0;
 							count++;
 						}
 					}
 					else {
 						int input_col = -pad + kernel_col * dilation;
 						for (int output_col = output_w; output_col; output_col--) {
 							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
 								float tmp = data[input_row * width + input_col];
 								*(col_buff++) = tmp;
 								count++;
 							}
 							else {
 								count++;
 								*(col_buff++) = 0;
 							}
 							input_col += stride;
 						}
 					}
 					input_row += stride;
 				}
 			}
 		}
 	}
 }

vector<float> my_conv_base(float* src, vector<float>& weight, int num_output, int kernel_size_height,
	int kernel_size_width, int outputwidth, int outputheight, int inputchannels, vector<float>& bias){
	const float* A = weight.data();
	float* B = src;
	int M = num_output;
	int N = outputheight*outputwidth;
	int K = inputchannels * kernel_size_height *kernel_size_width;
	int LDA = K;
	int LDB = N;
	const float alpha = 1;
	const float beta = 0;
	vector<float> ctmp(M*N, 0);
	float* C = ctmp.data();
	int LDC = N;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);

	const float* D = bias.data();
	vector<float> tmpvec(N, 1);//全为1,bias的维度是numoutput * 1
	float* tmp = tmpvec.data();
	//memset(tmp,1.0f,sizeof(tmp));
	const float beta2 = 1;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, 1, alpha, D, 1, tmp, N, beta2, C, N);
	//delete C;
	return ctmp;
}

float my_arg_max(Mat& region){
	float* data = region.ptr<float>(0);
	float tmp = data[0];
	for (int i = 0; i < region.rows; ++i){
		data = region.ptr<float>(i);
		for (int j = 0; j < region.cols; ++j){
			tmp = std::max(tmp, data[j]);
		}
	}
	return tmp;
}


void ClearVector(vector<Mat>& vt)
{
	/*	vector<Mat> vtTemp();*/
	vt.clear();
	vector<Mat>().swap(vt);
}

vector<Mat> my_conv_2d(int kernel_size, int stride, const vector<Mat>& src, int num_output, vector<float>& weights, vector<float>& bias, bool pad = true){
	//padding,为了简化，默认卷积核size为奇数，不存在取整问题，同时没有考虑dilation参数。
	int pad_size = 0;
	if (pad){
		pad_size = ((src[0].rows - 1)*stride + kernel_size - src[0].rows) / 2;
		for (int i = 0; i < src.size(); ++i){
			copyMakeBorder(src[i], src[i], pad_size, pad_size, pad_size, pad_size, BORDER_CONSTANT, (0));
		}
	}
	else{//在右方和下方补全
		for (int i = 0; i < src.size(); ++i){
			if ((src[i].rows - 1) % stride != 0)
				copyMakeBorder(src[i], src[i], 0, (src[i].rows - kernel_size) % stride, 0, 0, BORDER_CONSTANT, (0));
			if ((src[i].cols - 1) % stride != 0)
				copyMakeBorder(src[i], src[i], 0, 0, 0, (src[i].cols - kernel_size) % stride, BORDER_CONSTANT, (0));
		}

	}

	//定义输出维度
	int output_h = (src[0].rows - kernel_size) / stride + 1;
	int output_w = (src[0].cols - kernel_size) / stride + 1;
	int input_channels = src.size();


	//每个通道进行im2col，并添加到tmp
	// 		vector<Mat> im2col_tmp;
	// 		for (Mat mat : src){
	// 			im2col_tmp.emplace_back( im2col(kernel_size, stride, mat,1,output_h*output_w));
	// 		}
	vector<float> vectmp;
	for (Mat mat : src){
		Mat2vec(mat, vectmp);
	}
	float* im2col_tmp = vectmp.data();
	float* im2col_dst = new float[output_w*output_h*input_channels*kernel_size*kernel_size];
	im2col(im2col_tmp, input_channels, src[0].rows, src[0].cols, kernel_size, pad_size, stride, im2col_dst);
	
	//所有的im2col上下拼接，然后转为vec
	//Mat dst_tmp;
	//Mat* array = im2col_tmp.data();
	//vconcat(array, im2col_tmp.size(), dst_tmp);
	//ClearVector(im2col_tmp);

	//delete(array);
	// 		for (Mat mat : im2col_tmp){
	// 			mat.release();
	// 		}
	//dst_tmp.release();
	//im2col_tmp.swap(clear);
	//conv计算
	vector<float> output_tmp = my_conv_base(im2col_dst, weights, num_output, kernel_size, kernel_size, output_w, output_h, input_channels, bias);
	/*vector<float> output_tmp(output, output + sizeof(output) / sizeof(float));*/
	vector<Mat> dst = Vec2vecmat(output_tmp, num_output, output_h, output_w);
	//计算输出
	// 		for (Mat mat : dst){
	// 			mat = mat.t();
	// 		}
	return dst;
}

void p_relu(vector<Mat>& src, vector<float>& weights){
	//vector<Mat> dst = src;
	int nrows = src[0].rows;
	int ncols = src[0].cols;
	for (int k = 0; k < src.size(); ++k){
		for (int i = 0; i < nrows; ++i){
			float* data = src[k].ptr<float>(i);
			for (int j = 0; j < ncols; ++j){
				data[j] = std::max(data[j], 0.0f) + weights[k] * std::min(data[j], 0.0f);
			}

		}
	}
	//return dst;
}

vector<Mat> my_max_pool(int kernel_size, int stride, vector<Mat>& src, bool pad = false){
	//padding,为了简化，默认卷积核size为奇数，不存在取整问题，同时没有考虑dilation参数
	if (pad){
		int pad_size = ((src[0].rows - 1)*stride + kernel_size - src[0].rows) / 2;
		for (int i = 0; i < src.size(); ++i){
			copyMakeBorder(src[i], src[i], pad_size, pad_size, pad_size, pad_size, BORDER_CONSTANT, (0));
		}
	}
	else{//在右方和下方补全
		for (int i = 0; i < src.size(); ++i){
			if ((src[i].rows - kernel_size) % stride != 0)
				copyMakeBorder(src[i], src[i], 0, (src[i].rows - kernel_size) % stride, 0, 0, BORDER_CONSTANT, (-FLT_MAX));
			if ((src[i].cols - kernel_size) % stride != 0)
				copyMakeBorder(src[i], src[i], 0, 0, 0, (src[i].cols - kernel_size) % stride, BORDER_CONSTANT, (-FLT_MAX));
		}

	}


	//定义输出维度
	vector<Mat> dst;//(src.size(), Mat((src[0].rows - kernel_size) / stride + 1, (src[0].cols - kernel_size) / stride + 1, CV_32FC1, Scalar(0)));
	/*	dst_tmp.release();*/



	//计算输出
	for (int k = 0; k < src.size(); ++k){
		Mat tmp((src[0].rows - kernel_size) / stride + 1, (src[0].cols - kernel_size) / stride + 1, CV_32FC1, Scalar(0));
		for (int i = 0; i <= src[k].rows - ((src[k].rows - kernel_size) % stride) - kernel_size; i = i + stride){
			float* data = tmp.ptr<float>(i / stride);
			for (int j = 0; j <= src[k].cols - ((src[k].cols - kernel_size) % stride) - kernel_size; j = j + stride){
				//我发现rect，是列在先，行在后，也就是用x，y坐标来计算的。at方法是i行j列，按行列来的。
				Rect rect(j, i, kernel_size, kernel_size);
				Mat region = src[k](rect);
				data[j / stride] = my_arg_max(region);
				//tmp.at<float>(i / stride, j / stride) = my_arg_max(region);
			}
		}
		dst.emplace_back(tmp);
		// 		Mat tmp3 = dst[k];
		// 		Mat tmp = src[k];
	}

	return dst;
}

vector<float> my_fc(const vector<Mat>& src, vector<float>& weights, vector<float>& bias, int num_output){
	vector<float> src_vec;
	int kernel_size_width = src[0].cols;
	int kernel_size_height = src[0].rows;
	for (Mat mat : src){
		//vector <float> tmp = 
		Mat2vec(mat, src_vec);
		//src_vec.insert(src_vec.end(), tmp.begin(), tmp.end());
	}
	float* tmp = src_vec.data();
	vector<float> dst_tmp = my_conv_base(tmp, weights, num_output, kernel_size_height, kernel_size_width, 1, 1, src.size(), bias);
	//vector<Mat> dst = Vec2vecmat(dst_tmp, num_output, 1, 1);
	return dst_tmp;
}

vector<float> softmax(vector<float>& src){
	vector<float> dst(src.size());
	// 	for (int i = 0; i < src[0].rows; ++i){
	// 		for (int j = 0; j < src[0].cols; ++j){
	// 			float sum = 0;
	// 			for (int k = 0; k < )
	// 		}
	// 	}
	// 	
	float sum = 0;
	for (int k = 0; k < src.size(); ++k){
		sum += exp(src[k]);
	}
	for (int k = 0; k < src.size(); ++k){
		float prob = exp(src[k]) / sum;
		dst[k] = prob;
	}
	return dst;
}

struct size_pyra{
	float width;
	float height;
	size_pyra(int rows, int cols){
		this->width = cols;
		this->height = rows;
	}
	bool operator==(const size_pyra& n1)const
	{
		return (width == n1.width && height == n1.height);
	}
};

vector<size_pyra> pyramid(Mat& src, const int& minsize, const float& stepsize){
	vector<size_pyra> dst;
	size_pyra tmp(src.rows, src.cols);
	dst.emplace_back(tmp);
	while (tmp.height * stepsize >= minsize && tmp.width* stepsize >= minsize){
		tmp.height *= stepsize;
		tmp.width *= stepsize;
		dst.emplace_back(tmp);
	}

	return dst;
};

struct pred{
	float positive_prob;
	float negative_prob;
	float corner_ltx;
	float corner_lty;
	float corner_brx;
	float corner_bry;
	//float scale;
	vector<float> landmark;
	// 	float inputhetight;
	// 	float inputwidth;
 	bool operator==(const pred& n1)const
 	{
 		return (positive_prob == n1.positive_prob && negative_prob == n1.negative_prob
 			&& corner_ltx == n1.corner_ltx && corner_lty == n1.corner_lty && corner_brx == n1.corner_brx && corner_bry == n1.corner_bry
 			&& landmark == n1.landmark// && scale == n1.scale
 			//&& inputhetight == n1.inputhetight && inputwidth == n1.inputwidth
 			);
 	}
	float height() const{
		return corner_bry - corner_lty + 1;
	}
	float width()const{
		return corner_brx - corner_ltx + 1;
	}
	Rect box() const{
		return Rect(this->corner_ltx,this->corner_lty,this->width(),this->height());
	}
	Rect transbox() const{
		return Rect(this->corner_lty,this->corner_ltx,this->height(),this->width());
	}
	float area() const{
		return ((this->corner_brx - this->corner_ltx + 1) * (this->corner_bry - this->corner_lty + 1));
	}


};

float IOU_calc(const pred& pred_score, const pred& comparescore){
	float xmax = max(pred_score.corner_ltx, comparescore.corner_ltx);
	float ymax = max(pred_score.corner_lty, comparescore.corner_lty);
	float xmin = min(pred_score.corner_brx, comparescore.corner_brx);
	float ymin = min(pred_score.corner_bry, comparescore.corner_bry);

	float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
	float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
	float iou = uw * uh;
	if (iou == 0) return 0;

	//if (type == NMSType_IOUMin)
	//	return iou / min(a.area(), b.area());
	//else
	return iou / (pred_score.area() + comparescore.area() - iou);

}





void NMS(vector<pred>& pred_score, float iou_threshold){

	std::sort(pred_score.begin(), pred_score.end(), [](pred& a, pred& b){
		return a.positive_prob > b.positive_prob;
	});
	vector<pred> result;
	vector<int> flags(pred_score.size());
	for (int i = 0; i < pred_score.size(); ++i){
		if (flags[i] == 1)
			continue;
		result.emplace_back(pred_score[i]);
		for (int j = i + 1; j < pred_score.size(); ++j){
			float iou = IOU_calc(pred_score[i], pred_score[j]);
			if (iou > iou_threshold){
				flags[j] = 1;
			}
		}
	}
	pred_score = result;
}


class P_net{//输入只属于Pnet的参数
public:
	vector<vector<float>> weights_conv;
	vector<vector<float>> bias;
	vector<vector<float>> weights_pr;

	P_net(vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr){
		this->weights_conv = weights_conv;
		this->bias = bias;
		this->weights_pr = weights_pr;
	}

	vector<Mat> forward_bakcbone(const vector<Mat>& src, vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr){
		vector<Mat> x = my_conv_2d(3, 1, src, 10, weights_conv[0], bias[0], false);
		p_relu(x, weights_pr[0]);
		//Mat matrix( , x1[0].cols, CV_32F, x1);
		//auto x2 = p_relu(x1, weights);
		vector<Mat> x2 = my_max_pool(2, 2, x, false);
		//		ClearVector(x1);
		vector<Mat> x3 = my_conv_2d(3, 1, x2, 16, weights_conv[1], bias[1], false);
		p_relu(x3, weights_pr[1]);
		//		ClearVector(x2);
		auto x4 = my_conv_2d(3, 1, x3, 32, weights_conv[2], bias[2], false);
		p_relu(x4, weights_pr[2]);
		//auto x7 = p_relu(x6, weights);
		return x4;
	};

	vector<Mat> cls(const vector<Mat>& src, vector<vector<float>>& weights_conv, vector<vector<float>>& bias){
		vector<Mat> x = my_conv_2d(1, 1, src, 2, weights_conv[3], bias[3], false);
		return x;
	};

	vector<Mat> box(const vector<Mat>& src, vector<vector<float>>& weights_conv, vector<vector<float>>& bias){
		vector<Mat> x = my_conv_2d(1, 1, src, 4, weights_conv[4], weights_conv[4], false);
		return x;
	};

	vector<pred> box_confid(vector<Mat>& boxes, vector<Mat>& clses, float& threshold, float& scale){
		int inputsize = 12;
		int stride = 2;
		vector<pred> dst;
		for (int i = 0; i < boxes[0].rows; ++i){
			for (int j = 0; j < boxes[0].cols; ++j){
				//这里做softmax
				float negaprob = exp(clses[0].at<float>(i, j));
				float postiveprob = exp(clses[1].at<float>(i, j));
				float sum = negaprob + postiveprob;
				negaprob = negaprob / sum;
				postiveprob = postiveprob / sum;

				if (postiveprob >= threshold){
					pred tmp;
					tmp.positive_prob = postiveprob;
					tmp.negative_prob = negaprob;
					float dy = boxes[0].at<float>(i, j);
					float dx = boxes[1].at<float>(i, j);
					float db = boxes[2].at<float>(i, j);
					float dr = boxes[3].at<float>(i, j);
					tmp.corner_ltx = (j * stride + dx * inputsize) / scale;
					tmp.corner_lty = (i * stride + dy * inputsize) / scale;
					tmp.corner_brx = (j * stride + dr * inputsize + inputsize - 1) / scale;
					tmp.corner_bry = (i * stride + db * inputsize + inputsize - 1) / scale;
					float w = tmp.width();
					float h = tmp.height();
					float mxline = std::max(w, h);
					float cx = tmp.corner_ltx + w * 0.5;
					float cy = tmp.corner_lty + h * 0.5;

					tmp.corner_ltx = cx - mxline * 0.5;
					tmp.corner_lty = cy - mxline * 0.5;
					tmp.corner_brx = tmp.corner_ltx + mxline;
					tmp.corner_bry = tmp.corner_lty + mxline;

// 					tmp.corner_ltx = std::max(0.0f, std::min(float(cx- mxline * 0.5),float(boxes[0].cols) / scale - 1.0f));
// 					tmp.corner_lty = std::max(0.0f, std::min(float(cy - mxline * 0.5), float(boxes[0].rows) / scale - 1.0f));
// 					tmp.corner_brx = std::max(0.0f, std::min(float(tmp.corner_ltx + mxline),float(boxes[0].cols) / scale - 1.0f));
// 					tmp.corner_bry = std::max(0.0f, std::min(float(tmp.corner_lty + mxline), float(boxes[0].rows) / scale - 1.0f));
					//tmp.scale = scale;
					vector<float> x(10, 0);
					tmp.landmark = x;
					if (w * h > 0 )
						dst.emplace_back(tmp);
				}
			}
		}
		return dst;
	}
};


class R_net{//输入只属于Rnet的参数
public:
	vector<vector<float>> weights_conv;
	vector<vector<float>> bias;
	vector<vector<float>> weights_pr;
	vector<vector<float>> weights_fc;
	vector<vector<float>> bias_fc;

	R_net(vector<vector<float>>& weights_conv,
		vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		this->weights_conv = weights_conv;
		this->bias = bias;
		this->weights_pr = weights_pr;
		this->weights_fc = weights_fc;
		this->bias_fc = bias_fc;

	}

	vector<Mat> forward_bakcbone(vector<Mat>& src, vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		vector<Mat> x1 = my_conv_2d(3, 1, src, 28, weights_conv[0], bias[0], false);
		p_relu(x1, weights_pr[0]);
		vector<Mat> x2 = my_max_pool(3, 2, x1, false);
		//ClearVector(x1);
		vector<Mat> x3 = my_conv_2d(3, 1, x2, 48, weights_conv[1], bias[1], false);
		p_relu(x3, weights_pr[1]);
		//ClearVector(x2);
		vector<Mat> x4 = my_max_pool(3, 2, x3, false);
		//ClearVector(x3);
		vector<Mat> x5 = my_conv_2d(2, 1, x4, 64, weights_conv[2], bias[2], false);
		p_relu(x5, weights_pr[2]);
		//ClearVector(x4);
		vector<float> x = my_fc(x5, weights_fc[0], bias_fc[0], 128);
		vector<Mat> x6 = Vec2vecmat(x, 128, 1, 1);
		//vector<Mat> x7 = Vec2vecmat(x6, 128, 1, 1);
		//ClearVector(x5);
		p_relu(x6, weights_pr[3]);
		return x6;
	};

	vector<float> cls(vector<Mat>& src, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		auto cls = softmax(my_fc(src, weights_fc[1], bias_fc[1], 2));
		return cls;
	};

	vector<float> box(vector<Mat>& src, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		return  my_fc(src, weights_fc[2], bias_fc[2], 4);
	};




	void box_confid(vector<float>& box, vector<float>& cls, pred& pnet_out){
		//float width = pnet_out.corner_brx - pnet_out.corner_ltx + 1;
		//float height = pnet_out.corner_bry - pnet_out.corner_lty + 1;
		pnet_out.positive_prob = cls[1];
		pnet_out.negative_prob = cls[0];
		pnet_out.corner_lty = pnet_out.corner_lty + pnet_out.height() * box[0];
		pnet_out.corner_ltx = pnet_out.corner_ltx + pnet_out.width() * box[1];
		pnet_out.corner_bry = pnet_out.corner_bry + pnet_out.height() * box[2];
		pnet_out.corner_brx = pnet_out.corner_brx + pnet_out.width() * box[3];
		float w = pnet_out.width();
		float h = pnet_out.height();
		float mxline = std::max(w, h);
		float cx = pnet_out.corner_ltx + w * 0.5;
		float cy = pnet_out.corner_lty + h * 0.5;
		pnet_out.corner_ltx = cx - mxline * 0.5;
		pnet_out.corner_lty = cy - mxline * 0.5;
		pnet_out.corner_brx = pnet_out.corner_ltx + mxline;
		pnet_out.corner_bry = pnet_out.corner_lty + mxline;
	}

};

class O_net{//输入只属于Onet的参数
private:
	vector<vector<float>> weights_conv;
	vector<vector<float>> bias;
	vector<vector<float>> weights_pr;
	vector<vector<float>> weights_fc;
	vector<vector<float>> bias_fc;
public:
	O_net(vector<vector<float>>& weights_conv,
		vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		this->weights_conv = weights_conv;
		this->bias = bias;
		this->weights_pr = weights_pr;
		this->weights_fc = weights_fc;
		this->bias_fc = bias_fc;
	}
	vector<Mat> forward_bakcbone(vector<Mat>& src, vector<vector<float>>& weights_conv,
		vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		vector<Mat> x1 = my_conv_2d(3, 1, src, 32, weights_conv[0], bias[0], false);
		p_relu(x1, weights_pr[0]);
		vector<Mat> x2 = my_max_pool(3, 2, x1, false);
		x1.shrink_to_fit();
		vector<Mat> x3 = my_conv_2d(3, 1, x2, 64, weights_conv[1], bias[1], false);
		p_relu(x3, weights_pr[1]);
		x2.shrink_to_fit();
		vector<Mat> x4 = my_max_pool(3, 2, x3, false);
		x3.shrink_to_fit();
		vector<Mat> x5 = my_conv_2d(3, 1, x4, 64, weights_conv[2], bias[2], false);
		p_relu(x5, weights_pr[2]);
		x4.shrink_to_fit();
		vector<Mat> x6 = my_max_pool(3, 2, x5, false);
		x5.shrink_to_fit();
		vector<Mat> x7 = my_conv_2d(2, 1, x6, 128, weights_conv[3], bias[3], false);
		p_relu(x7, weights_pr[3]);
		x6.shrink_to_fit();
		vector<Mat> x8 = Vec2vecmat(my_fc(x7, weights_fc[0], bias_fc[0], 256), 256, 1, 1);
		x7.shrink_to_fit();
		p_relu(x8, weights_pr[4]);
		return x8;
	};

	vector<float> cls(vector<Mat>& src, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		auto cls = softmax(my_fc(src, weights_fc[1], bias_fc[1], 2));
		return cls;
	};
	vector<float> box(vector<Mat>& src, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		return  my_fc(src, weights_fc[2], bias_fc[2], 4);
	};
	vector<float> landmark(vector<Mat>& src, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		return my_fc(src, weights_fc[3], bias_fc[3], 10);
	}

	void box_confid(vector<float>& box, vector<float>& cls, vector<float>& landmark, pred& rnet_output){
		//float width = rnet_output.corner_brx - rnet_output.corner_ltx + 1;
		//float height = rnet_output.corner_bry - rnet_output.corner_lty + 1;
		rnet_output.positive_prob = cls[1];
		rnet_output.negative_prob = cls[0];
		rnet_output.corner_lty = rnet_output.corner_lty + rnet_output.height() * box[0];
		rnet_output.corner_ltx = rnet_output.corner_ltx + rnet_output.width() * box[1];
		rnet_output.corner_bry = rnet_output.corner_bry + rnet_output.height() * box[2];
		rnet_output.corner_brx = rnet_output.corner_brx + rnet_output.width() * box[3];
		float w = rnet_output.width();
		float h = rnet_output.height();
		float mxline = std::max(w, h);
		float cx = rnet_output.corner_ltx + w * 0.5;
		float cy = rnet_output.corner_lty + h * 0.5;
		rnet_output.corner_ltx = cx - mxline * 0.5;
		rnet_output.corner_lty = cy - mxline * 0.5;
		rnet_output.corner_brx = rnet_output.corner_ltx + mxline;
		rnet_output.corner_bry = rnet_output.corner_lty + mxline;
		rnet_output.landmark = landmark;
		//			dst[i].corner_ltx = box[i][2];
		//return dst;
	}
};

class mtcnn{
private:
	int minsize;
	float stepsize;
	float iou_threshold;
	float threshold_1;
	float threshold_2;
	float threshold_3;
	int Rnet_input_size;
	int Onet_input_size;
	Mat src_pic;
public:
	mtcnn(int minsize, float stepsize, float iou_threshold, float threshold_1, float threshold_2, float threshold_3, Mat src_pic, int Rnet_input_size, int Onet_input_size){
		this->minsize = minsize;
		this->stepsize = stepsize;
		this->iou_threshold = iou_threshold;
		this->threshold_1 = threshold_1;
		this->threshold_2 = threshold_2;
		this->threshold_3 = threshold_3;
		this->Rnet_input_size = Rnet_input_size;
		this->Onet_input_size = Onet_input_size;
		this->src_pic = src_pic;
	};

	vector<pred> forward_Pnet(Mat& src, vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr){
		auto pnet = P_net(weights_conv, bias, weights_pr);
		vector<size_pyra> pyra = pyramid(src, this->minsize, this->stepsize);
		vector<pred> dst;
		vector<Mat> vecmat_tmp(3);
		for (int i = 0; i < pyra.size(); ++i){
			vecmat_tmp.clear();
			resize(src, src, Size(pyra[i].width, pyra[i].height));
			split(src, vecmat_tmp);
			vector<Mat> x = pnet.forward_bakcbone(vecmat_tmp, pnet.weights_conv, pnet.bias, pnet.weights_pr);
			//ClearVector(vecmat_tmp);
			auto cls_pred = pnet.cls(x, pnet.weights_conv, pnet.bias);
			auto box_pred = pnet.box(x, pnet.weights_conv, pnet.bias);
			float scale = float(pyra[i].height) / float(this->src_pic.rows);
			vector<pred> pred_score = pnet.box_confid(box_pred, cls_pred, this->threshold_1, scale);
			//NMS(pred_score,this->iou_threshold,this->threshold_1);
			NMS(pred_score, this->iou_threshold);
			dst.insert(dst.end(), pred_score.begin(), pred_score.end());
		}
		// 		vector<Mat> for_rnet_tmp;
		// 		vector<vector<Mat>> for_rnet;
		NMS(dst, this->iou_threshold);

		return dst;
	};

	vector<pred> forward_Rnet(vector<pred>& pnet_output, vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		auto rnet = R_net(weights_conv, bias, weights_pr, weights_fc, bias_fc);
		vector<Mat> for_rnet_tmp(3);
		vector<pred> Rnet_output;
		//vector<vector<Mat>> for_rnet;
		for (int i = 0; i < pnet_output.size(); ++i){
// 			if (pnet_output[i].corner_ltx + pnet_output[i].width() > src_pic.cols || pnet_output[i].corner_lty + (pnet_output[i].corner_bry - pnet_output[i].corner_lty) + 1 > src_pic.rows
// 				|| pnet_output[i].corner_brx > src_pic.cols || pnet_output[i].corner_bry > src_pic.rows || pnet_output[i].corner_ltx >= this->src_pic.cols
// 				|| pnet_output[i].corner_lty >= this->src_pic.rows || 
// 				pnet_output[i].width() < 0)
// 			{
			pnet_output[i].corner_ltx = std::max(0.0f, std::min(pnet_output[i].corner_ltx, float(this->src_pic.cols)));
			pnet_output[i].corner_lty = std::max(0.0f, std::min(pnet_output[i].corner_lty, float(this->src_pic.rows)));
			pnet_output[i].corner_brx = std::max(0.0f, std::min(pnet_output[i].corner_brx, float(this->src_pic.cols)));
			pnet_output[i].corner_bry = std::max(0.0f, std::min(pnet_output[i].corner_bry, float(this->src_pic.rows)));
			if (pnet_output[i].width() < 0 || pnet_output[i].height() < 0)
					continue;
			//}

			//传进来的图有做transpose，但我截图的地方是没做transpose的，所以用point(x,y)
			Mat region = this->src_pic(pnet_output[i].box());
			resize(region, region, Size(this->Rnet_input_size, this->Rnet_input_size));
			split(region, for_rnet_tmp);

			vector<Mat> x = rnet.forward_bakcbone(for_rnet_tmp, rnet.weights_conv, bias, weights_pr, weights_fc, bias_fc);
			vector<float> cls_pred = rnet.cls(x, weights_fc, bias_fc);
			if (cls_pred[1] < this->threshold_2){
				// 				pnet_output.erase(pnet_output.begin() + i);
				// 				//for_rnet.erase(for_rnet.begin() + i);
				// 				i--;
				for_rnet_tmp.clear();
				//if (i == pnet_output.size() - 1)
				//	break;
				//else 
					continue;
			}
			else {
				vector<float> box_pred = rnet.box(x, weights_fc, bias_fc);
				rnet.box_confid(box_pred, cls_pred, pnet_output[i]);
				if(pnet_output[i].width() * pnet_output[i].height() > 0)
					Rnet_output.emplace_back(pnet_output[i]);
			};
			//for_rnet.emplace_back(for_rnet_tmp);
			for_rnet_tmp.clear();
		}
		NMS(Rnet_output, this->iou_threshold);
		return Rnet_output;
	};

	vector<pred> forward_Onet(vector<pred>& rnetoutput, vector<vector<float>>& weights_conv, vector<vector<float>>& bias, vector<vector<float>>& weights_pr, vector<vector<float>>& weights_fc, vector<vector<float>>& bias_fc){
		auto onet = O_net(weights_conv, bias, weights_pr, weights_fc, bias_fc);
		vector<Mat> for_onet_tmp(3);
		//vector<vector<Mat>> for_onet;
		vector<pred> Onet_output;
 		for (int i = 0; i < rnetoutput.size(); ++i){
// 			if (rnetoutput[i].corner_ltx + (rnetoutput[i].corner_brx - rnetoutput[i].corner_ltx) + 1 > src_pic.cols || rnetoutput[i].corner_lty + (rnetoutput[i].corner_bry - rnetoutput[i].corner_lty) + 1 > src_pic.rows
// 				|| rnetoutput[i].corner_brx > src_pic.cols || rnetoutput[i].corner_bry > src_pic.rows || rnetoutput[i].width() < 0)
// 			{
				rnetoutput[i].corner_ltx = std::max(0.0f, std::min(rnetoutput[i].corner_ltx, float(this->src_pic.cols)));
				rnetoutput[i].corner_lty = std::max(0.0f, std::min(rnetoutput[i].corner_lty, float(this->src_pic.rows)));
				rnetoutput[i].corner_brx = std::max(0.0f, std::min(rnetoutput[i].corner_brx, float(this->src_pic.cols)));
				rnetoutput[i].corner_bry = std::max(0.0f, std::min(rnetoutput[i].corner_bry, float(this->src_pic.rows)));
				if (rnetoutput[i].width() < 0 || rnetoutput[i].height() < 0)
					continue;
			//}

			Mat region = this->src_pic(rnetoutput[i].box());
			resize(region, region, Size(this->Onet_input_size, this->Onet_input_size));
			split(region, for_onet_tmp);
			//for_onet_tmp.emplace_back(region);
			//}
			vector<Mat> x = onet.forward_bakcbone(for_onet_tmp, weights_conv, bias, weights_pr, weights_fc, bias_fc);
			
			vector<float> cls_pred = onet.cls(x, weights_fc, bias_fc);
			if (cls_pred[1] < this->threshold_2){
				// 				rnetoutput.erase(rnetoutput.begin() + i);
				// 				//for_onet.erase(for_onet.begin() + i);
				// 				i--;
				for_onet_tmp.clear();
				continue;
			}
			else {
				vector<float> box_pred = onet.box(x, weights_fc, bias_fc);
				auto landmark = onet.landmark(x, weights_fc, bias_fc);
				onet.box_confid(box_pred, cls_pred, landmark, rnetoutput[i]);
				if (rnetoutput[i].width() * rnetoutput[i].height() > 0)
					Onet_output.emplace_back(rnetoutput[i]);
			};
			//for_onet.emplace_back(for_onet_tmp);
			for_onet_tmp.clear();
		}

		NMS(Onet_output, this->iou_threshold);
		return Onet_output;

	};



};

int main(){
	Mat im = imread("./workspace/cmj2.jpg");
	//resize(im, im, Size(1200, 900));
	Mat inputimage;
	cvtColor(im, inputimage, CV_BGR2RGB);
	inputimage = inputimage.t();

	inputimage.convertTo(inputimage, CV_32F, 1 / 127.5, -1.0);

	vector<vector<vector<float>>> weights_conv, bias, weights_pr, weights_fc, bias_fc;
	auto net = mtcnn(12, 0.709, 0.5, 0.7, 0.8, 0.9, inputimage, 24, 48);
	storemodel(weights_conv, bias, weights_pr, weights_fc, bias_fc);
	double tick = getTickCount();
	vector<pred> pnet_output = net.forward_Pnet(inputimage, weights_conv[0], bias[0], weights_pr[0]);
	vector<pred> rnet_output = net.forward_Rnet(pnet_output, weights_conv[1], bias[1], weights_pr[1], weights_fc[1], bias_fc[1]);
	vector<pred> onet_output = net.forward_Onet(rnet_output, weights_conv[2], bias[2], weights_pr[2], weights_fc[2], bias_fc[2]);

	tick = (getTickCount() - tick) / getTickFrequency() * 1000;
	printf("耗时：%.2f ms\n", tick);

	for (pred pd : onet_output){

		rectangle(im, pd.transbox(), Scalar(0, 255), 2);
		for (int i = 0; i < pd.landmark.size() / 2; ++i){
			float y = pd.landmark[i] * pd.height() + pd.corner_lty;
			float x = pd.landmark[i + 5] * pd.width() + pd.corner_ltx;
			circle(im, Point2f(y, x),1, Scalar(0, 0, 255), -1);
		}
	}	
	imshow("demo", im);
	waitKey();
	return 0;
}
