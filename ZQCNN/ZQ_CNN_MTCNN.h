﻿#ifndef _ZQ_CNN_MTCNN_H_
#define _ZQ_CNN_MTCNN_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <omp.h>
namespace ZQ
{
	class ZQ_CNN_MTCNN
	{
	public:
		using string = std::string;
		ZQ_CNN_MTCNN()
		{
			min_size = 60;
			thresh[0] = 0.6;
			thresh[1] = 0.7;
			thresh[2] = 0.7;
			nms_thresh[0] = 0.6;
			nms_thresh[1] = 0.7;
			nms_thresh[2] = 0.7;
			width = 0;
			height = 0;
			factor = 0.709;
			pnet_overlap_thresh_count = 4;
			pnet_size = 12;
			pnet_stride = 2;
			special_handle_very_big_face = false;
			force_run_pnet_multithread = false;
			show_debug_info = false;
			limit_r_num = 0;
			limit_o_num = 0;
			limit_l_num = 0;
		}
		~ZQ_CNN_MTCNN()
		{

		}

	private:
#if __ARM_NEON
		const int BATCH_SIZE = 16;
#else
		const int BATCH_SIZE = 64;
#endif
		std::vector<ZQ_CNN_Net> pnet, rnet, onet, lnet;
		bool has_lnet;
		int thread_num;
		float thresh[3], nms_thresh[3];
		int min_size;
		int width, height;
		float factor;
		int pnet_overlap_thresh_count;
		int pnet_size;
		int pnet_stride;
		int rnet_size;
		int onet_size;
		int lnet_size;
		bool special_handle_very_big_face;
		bool do_landmark;
		float early_accept_thresh;
		float nms_thresh_per_scale;
		bool force_run_pnet_multithread;
		std::vector<float> scales;
		std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> pnet_images;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit ori_input, rnet_image, onet_image;
		bool show_debug_info;
		int limit_r_num;
		int limit_o_num;
		int limit_l_num;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		void SetLimit(int limit_r = 0, int limit_o = 0, int limit_l = 0) 
		{
			limit_r_num = limit_r;
			limit_o_num = limit_o;
			limit_l_num = limit_l;
		}

		bool Init(const string& pnet_param, const string& pnet_model, const string& rnet_param, const string& rnet_model,
			const string& onet_param, const string& onet_model, int thread_num = 1, 
			bool has_lnet = false, const string& lnet_param = "", const std::string& lnet_model = "")
		{
			if (thread_num < 1)
				force_run_pnet_multithread = true;
			else
				force_run_pnet_multithread = false;
			thread_num = __max(1, thread_num);
			pnet.resize(thread_num);
			rnet.resize(thread_num);
			onet.resize(thread_num);
			this->has_lnet = has_lnet;
			if (has_lnet)
			{
				lnet.resize(thread_num);
			}
			bool ret = true;
			for (int i = 0; i < thread_num; i++)
			{
				ret = pnet[i].LoadFrom(pnet_param, pnet_model,true,1e-9, true) 
					&& rnet[i].LoadFrom(rnet_param, rnet_model, true, 1e-9, true)
					&& onet[i].LoadFrom(onet_param, onet_model, true, 1e-9, true);
				if (has_lnet && ret)
					ret = lnet[i].LoadFrom(lnet_param, lnet_model, true, 1e-9, true);
				if (!ret)
					break;
			}
			if (!ret)
			{
				pnet.clear();
				rnet.clear();
				onet.clear();
				if (has_lnet)
					lnet.clear();
				this->thread_num = 0;
			}
			else
				this->thread_num = thread_num;
			if (show_debug_info)
			{
				printf("rnet = %.2f M, onet = %.2f M\n", rnet[0].GetNumOfMulAdd() / (1024.0*1024.0),
					onet[0].GetNumOfMulAdd() / (1024.0*1024.0));
				if (has_lnet)
					printf("lnet = %.2f M\n", lnet[0].GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			rnet[0].GetInputDim(C, H, W);
			rnet_size = H;
			onet[0].GetInputDim(C, H, W);
			onet_size = H;
			if (has_lnet)
			{
				lnet[0].GetInputDim(C, H, W);
				lnet_size = H;
			}
			return ret;
		}

		bool InitFromBuffer(
			const char* pnet_param, __int64 pnet_param_len, const char* pnet_model, __int64 pnet_model_len,
			const char* rnet_param, __int64 rnet_param_len, const char* rnet_model, __int64 rnet_model_len,
			const char* onet_param, __int64 onet_param_len, const char* onet_model, __int64 onet_model_len,
			int thread_num = 1, bool has_lnet = false, 
			const char* lnet_param = 0, __int64 lnet_param_len = 0, const char* lnet_model = 0, __int64 lnet_model_len = 0)
		{
			if (thread_num < 1)
				force_run_pnet_multithread = true;
			else
				force_run_pnet_multithread = false;
			thread_num = __max(1, thread_num);
			pnet.resize(thread_num);
			rnet.resize(thread_num);
			onet.resize(thread_num);
			this->has_lnet = has_lnet;
			if(has_lnet)
				lnet.resize(thread_num);
			bool ret = true;
			for (int i = 0; i < thread_num; i++)
			{
				ret = pnet[i].LoadFromBuffer(pnet_param, pnet_param_len,pnet_model,pnet_model_len, true, 1e-9, true)
					&& rnet[i].LoadFromBuffer(rnet_param, rnet_param_len, rnet_model, rnet_model_len, true, 1e-9, true)
					&& onet[i].LoadFromBuffer(onet_param, onet_param_len, onet_model, onet_model_len, true, 1e-9, true);
				if (has_lnet && ret)
					ret = lnet[i].LoadFromBuffer(lnet_param, lnet_param_len, lnet_model, lnet_model_len, true, 1e-9, true);
				if (!ret)
					break;
			}
			if (!ret)
			{
				pnet.clear();
				rnet.clear();
				onet.clear();
				if (has_lnet)
					lnet.clear();
				this->thread_num = 0;
			}
			else
				this->thread_num = thread_num;
			if (show_debug_info)
			{
				printf("rnet = %.1f M, onet = %.1f M\n", rnet[0].GetNumOfMulAdd() / (1024.0*1024.0),
					onet[0].GetNumOfMulAdd() / (1024.0*1024.0));
				if (has_lnet)
					printf("lnet = %.1f M\n", lnet[0].GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			rnet[0].GetInputDim(C, H, W);
			rnet_size = H;
			onet[0].GetInputDim(C, H, W);
			onet_size = H;
			if (has_lnet) {
				lnet [0].GetInputDim (C, H, W);
				lnet_size = H;
			}
			return ret;
		}

		void SetPara(int w, int h, int min_face_size = 60, float pthresh = 0.6, float rthresh = 0.7, float othresh = 0.7,
			float nms_pthresh = 0.6, float nms_rthresh = 0.7, float nms_othresh = 0.7, float scale_factor = 0.709, 
			int pnet_overlap_thresh_count = 4, int pnet_size = 12, int pnet_stride = 2, bool special_handle_very_big_face = false, 
			bool do_landmark = true, float early_accept_thresh = 1.00)
		{
			min_size = __max(pnet_size, min_face_size);
			thresh[0] = __max(0.1, pthresh); thresh[1] = __max(0.1, rthresh); thresh[2] = __max(0.1, othresh);
			nms_thresh[0] = __max(0.1, nms_pthresh); nms_thresh[1] = __max(0.1, nms_rthresh); nms_thresh[2] = __max(0.1, nms_othresh);
			scale_factor = __max(0.5, __min(0.97, scale_factor));
			this->pnet_overlap_thresh_count = __max(0, pnet_overlap_thresh_count);
			this->pnet_size = pnet_size;
			this->pnet_stride = pnet_stride;
			this->special_handle_very_big_face = special_handle_very_big_face;
			this->do_landmark = do_landmark;
			this->early_accept_thresh = early_accept_thresh;
			if (pnet_size == 20 && pnet_stride == 4)
				nms_thresh_per_scale = 0.45;
			else
				nms_thresh_per_scale = 0.495;
			if (width != w || height != h || factor != scale_factor)
			{
				scales.clear();
				pnet_images.clear();

				width = w; height = h;
				float minside = __min(width, height);
				int MIN_DET_SIZE = pnet_size;
				float m = (float)MIN_DET_SIZE / min_size;
				minside *= m;
				while (minside > MIN_DET_SIZE)
				{
					scales.push_back(m);
					minside *= factor;
					m *= factor;
				}
				minside = __min(width, height);
				int count = scales.size();
				for (int i = scales.size() - 1; i >= 0; i--)
				{
					if (ceil(scales[i] * minside) <= pnet_size)
					{
						count--;
					}
				}
				if (special_handle_very_big_face)
				{
					if (count > 2)
						count--;

					scales.resize(count);
					if (count > 0)
					{
						float last_size = ceil(scales[count - 1] * minside);
						for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1; tmp_size -= 2)
						{
							scales.push_back((float)tmp_size / minside);
							count++;
						}
					}
					
					scales.push_back((float)pnet_size / minside);
					count++;
				}
				else
				{
					scales.push_back((float)pnet_size / minside);
					count++;
				}

				pnet_images.resize(count);
			}
		}

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& results)
		{
			double t1 = omp_get_wtime();

			if (width != _width || height != _height)
				return false;
			if (!ori_input.ConvertFromBGR(bgr_img, width, height, _widthStep))
				return false;
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("convert cost: %.3f ms\n", 1000 * (t2 - t1));
			return Find(ori_input, results);
		}

		bool Find106(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox106>& results)
		{
			double t1 = omp_get_wtime();

			if (width != _width || height != _height)
				return false;
			if (!ori_input.ConvertFromBGR(bgr_img, width, height, _widthStep))
				return false;
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("convert cost: %.3f ms\n", 1000 * (t2 - t1));
			return Find106(ori_input, results);
		}

		bool Find(ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& results)
		{
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox> firstBbox, secondBbox, thirdBbox;
			if (!_Pnet_stage(input, firstBbox))
				return false;
			//results = firstBbox;
			//return true;
			if (limit_r_num > 0)
			{
				_select(firstBbox, limit_r_num, input.GetW(), input.GetH());
			}

			double t2 = omp_get_wtime();
			if (!_Rnet_stage(input, firstBbox, secondBbox))
				return false;
			//results = secondBbox;
			//return true;

			if (limit_o_num > 0)
			{
				_select(secondBbox, limit_o_num, input.GetW(), input.GetH());
			}

			if (!has_lnet || !do_landmark)
			{
				double t3 = omp_get_wtime();
				if (!_Onet_stage(input, secondBbox, results))
					return false;

				double t4 = omp_get_wtime();
				if (show_debug_info)
				{
					printf("final found num: %d\n", (int)results.size());
					printf("total cost: %.3f ms (P: %.3f ms, R: %.3f ms, O: %.3f ms)\n",
						1000 * (t4 - t1), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3));
				}
			}
			else
			{
				double t3 = omp_get_wtime();
				if (!_Onet_stage(input, secondBbox, thirdBbox))
					return false;

				if (limit_l_num > 0)
				{
					_select(thirdBbox, limit_l_num, input.GetW(), input.GetH());
				}

				double t4 = omp_get_wtime();

				if (!_Lnet_stage(input, thirdBbox, results))
					return false;

				double t5 = omp_get_wtime();
				if (show_debug_info)
				{
					printf("final found num: %d\n", (int)results.size());
					printf("total cost: %.3f ms (P: %.3f ms, R: %.3f ms, O: %.3f ms, L: %.3f ms)\n",
						1000 * (t5 - t1), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
				}
			}
			
			return true;
		}

		bool Find106(ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox106>& results)
		{
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox> firstBbox, secondBbox, thirdBbox;
			if (!_Pnet_stage(input, firstBbox))
				return false;
			//results = firstBbox;
			//return true;
			if (limit_r_num > 0)
			{
				_select(firstBbox, limit_r_num, input.GetW(), input.GetH());
			}
			double t2 = omp_get_wtime();
			if (!_Rnet_stage(input, firstBbox, secondBbox))
				return false;
			//results = secondBbox;
			//return true;
			if (limit_o_num > 0)
			{
				_select(secondBbox, limit_o_num, input.GetW(), input.GetH());
			}
			if (!has_lnet || !do_landmark)
			{
				return false;
			}
			double t3 = omp_get_wtime();
			if (!_Onet_stage(input, secondBbox, thirdBbox))
				return false;

			if (limit_l_num > 0)
			{
				_select(thirdBbox, limit_l_num, input.GetW(), input.GetH());
			}
			double t4 = omp_get_wtime();

			if (!_Lnet106_stage(input, thirdBbox, results))
				return false;

			double t5 = omp_get_wtime();
			if (show_debug_info)
			{
				printf("final found num: %d\n", (int)results.size());
				printf("total cost: %.3f ms (P: %.3f ms, R: %.3f ms, O: %.3f ms, L: %.3f ms)\n",
					1000 * (t5 - t1), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3), 1000 * (t5 - t4));
			}
			
			return true;
		}

	private:
		void _compute_Pnet_single_thread(ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<std::vector<float> >& maps, 
			std::vector<int>& mapH, std::vector<int>& mapW)
		{
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			maps.resize(scale_num);
			for (int i = 0; i < scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}

			for (int i = 0; i < scale_num; i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				float cur_scale_x = (float)width / changedW;
				float cur_scale_y = (float)height / changedH;
				double t10 = omp_get_wtime();
				if (scales[i] != 1)
				{
					input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
				}

				double t11 = omp_get_wtime();
				if (scales[i] != 1)
					pnet[0].Forward(pnet_images[i]);
				else
					pnet[0].Forward(input);
				double t12 = omp_get_wtime();
				if (show_debug_info)
					printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n",
						i, changedW, changedH, 1000 * (t11 - t10), 1000 * (t12 - t11));
				const ZQ_CNN_Tensor4D* score = pnet[0].GetBlobByName("prob1");
				//score p
				int scoreH = score->GetH();
				int scoreW = score->GetW();
				int scorePixStep = score->GetPixelStep();
				const float *p = score->GetFirstPixelPtr() + 1;
				for (int row = 0; row < scoreH; row++)
				{
					for (int col = 0; col < scoreW; col++)
					{
						if(row < mapH[i] && col < mapW[i])
							maps[i][row*mapW[i] + col] = *p;
						p += scorePixStep;
					}
				}
			}
		}
		void _compute_Pnet_multi_thread(ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<std::vector<float> >& maps,
			std::vector<int>& mapH, std::vector<int>& mapW)
		{
			if (thread_num <= 1)
			{
				for (int i = 0; i < scales.size(); i++)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales[i] != 1)
					{
						input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, 1)
				for (int i = 0; i < scales.size(); i++)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales[i] != 1)
					{
						input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
					}
				}
			}
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			maps.resize(scale_num);
			for (int i = 0; i < scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}
			
			std::vector<int> task_rect_off_x;
			std::vector<int> task_rect_off_y;
			std::vector<int> task_rect_width;
			std::vector<int> task_rect_height;
			std::vector<float> task_scale;
			std::vector<int> task_scale_id;

			int stride = pnet_stride;
			const int block_size = 64 * stride;
			int cellsize = pnet_size;
			int border_size = cellsize - stride;
			int overlap_border_size = cellsize / stride;
			int jump_size = block_size - border_size;
			for (int i = 0; i < scales.size(); i++)
			{
				int changeH = (int)ceil(height*scales[i]);
				int changeW = (int)ceil(width*scales[i]);
				if (changeH < pnet_size || changeW < pnet_size)
					continue;
				int block_H_num = 0;
				int block_W_num = 0;
				int start = 0;
				while (start < changeH)
				{
					block_H_num++;
					if (start + block_size >= changeH)
						break;
					start += jump_size;
				}
				start = 0;
				while (start < changeW)
				{
					block_W_num++;
					if (start + block_size >= changeW)
						break;
					start += jump_size;
				}
				for (int s = 0; s < block_H_num; s++)
				{
					for (int t = 0; t < block_W_num; t++)
					{
						int rect_off_x = t * jump_size;
						int rect_off_y = s * jump_size;
						int rect_width = __min(changeW, rect_off_x + block_size) - rect_off_x;
						int rect_height = __min(changeH, rect_off_y + block_size) - rect_off_y;
						if (rect_width >= cellsize && rect_height >= cellsize)
						{
							task_rect_off_x.push_back(rect_off_x);
							task_rect_off_y.push_back(rect_off_y);
							task_rect_width.push_back(rect_width);
							task_rect_height.push_back(rect_height);
							task_scale.push_back(scales[i]);
							task_scale_id.push_back(i);
						}
					}
				}
			}

			//
			int task_num = task_scale.size();
			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_pnet_images(thread_num);

			if (thread_num <= 1)
			{
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = 0;// omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
					if (scale_id == 0 && scales[0] == 1)
					{
						if (!input.ROI(task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
							continue;
					}
					else
					{
						if (!pnet_images[scale_id].ROI(task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
							continue;
					}

					if (!pnet[thread_id].Forward(task_pnet_images[thread_id]))
						continue;
					const ZQ_CNN_Tensor4D* score = pnet[thread_id].GetBlobByName("prob1");

					int task_count = 0;
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p += scorePixStep;
						}
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num)
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
					if (scale_id == 0 && scales[0] == 1)
					{
						if (!input.ROI(task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
							continue;
					}
					else
					{
						if (!pnet_images[scale_id].ROI(task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
							continue;
					}

					if (!pnet[thread_id].Forward(task_pnet_images[thread_id]))
						continue;
					const ZQ_CNN_Tensor4D* score = pnet[thread_id].GetBlobByName("prob1");

					int task_count = 0;
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p += scorePixStep;
						}
					}
				}
			}
		}

		bool _Pnet_stage(ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& firstBbox)
		{
			if (thread_num <= 0)
				return false;

			double t1 = omp_get_wtime();
			firstBbox.clear();
			std::vector<std::vector<float> > maps;
			std::vector<int> mapH;
			std::vector<int> mapW;
			if (thread_num == 1 && !force_run_pnet_multithread)
			{
				pnet[0].TurnOffShowDebugInfo();
				//pnet[0].TurnOnShowDebugInfo();
				_compute_Pnet_single_thread(input, maps, mapH, mapW);
			}
			else
			{
				_compute_Pnet_multi_thread(input, maps, mapH, mapW);
			}
			ZQ_CNN_OrderScore order;
			std::vector<std::vector<ZQ_CNN_BBox> > bounding_boxes(scales.size());
			std::vector<std::vector<ZQ_CNN_OrderScore> > bounding_scores(scales.size());
			const int block_size = 32;
			int stride = pnet_stride;
			int cellsize = pnet_size;
			int border_size = cellsize / stride;
			
			for (int i = 0; i < maps.size(); i++)
			{
				double t13 = omp_get_wtime();
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				float cur_scale_x = (float)width / changedW;
				float cur_scale_y = (float)height / changedH;
				
				int count = 0;
				//score p
				int scoreH = mapH[i];
				int scoreW = mapW[i];
				const float *p = &maps[i][0];
				if (scoreW <= block_size && scoreH < block_size)
				{
				
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (*p > thresh[0])
							{
								bbox.score = *p;
								order.score = *p;
								order.oriOrder = count;
								bbox.row1 = stride*row;
								bbox.col1 = stride*col;
								bbox.row2 = stride*row + cellsize;
								bbox.col2 = stride*col + cellsize;
								bbox.exist = true;
								bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
								bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
									&& (col >= border_size && col < scoreW - border_size);
								bounding_boxes[i].push_back(bbox);
								bounding_scores[i].push_back(order);
								count++;
							}
							p ++;
						}
					}
					int before_count = bounding_boxes[i].size();
					ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
					int after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}
				else
				{
					int before_count = 0, after_count = 0;
					int block_H_num = __max(1, scoreH / block_size);
					int block_W_num = __max(1, scoreW / block_size);
					int block_num = block_H_num*block_W_num;
					int width_per_block = scoreW / block_W_num;
					int height_per_block = scoreH / block_H_num;
					std::vector<std::vector<ZQ_CNN_BBox> > tmp_bounding_boxes(block_num);
					std::vector<std::vector<ZQ_CNN_OrderScore> > tmp_bounding_scores(block_num);
					std::vector<int> block_start_w(block_num), block_end_w(block_num);
					std::vector<int> block_start_h(block_num), block_end_h(block_num);
					for (int bh = 0; bh < block_H_num; bh++)
					{
						for (int bw = 0; bw < block_W_num; bw++)
						{
							int bb = bh * block_W_num + bw;
							block_start_w[bb] = (bw == 0) ? 0 : (bw*width_per_block - border_size);
							block_end_w[bb] = (bw == block_num - 1) ? scoreW : ((bw + 1)*width_per_block);
							block_start_h[bb] = (bh == 0) ? 0 : (bh*height_per_block - border_size);
							block_end_h[bb] = (bh == block_num - 1) ? scoreH : ((bh + 1)*height_per_block);
						}
					}
					int chunk_size = 1;// ceil((float)block_num / thread_num);
					if (thread_num <= 1)
					{
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							ZQ_CNN_BBoxUtils::_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}
					else
					{
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num)
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								const float* p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							ZQ_CNN_BBoxUtils::_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}

					count = 0;
					for (int bb = 0; bb < block_num; bb++)
					{
						std::vector<ZQ_CNN_BBox>::iterator it = tmp_bounding_boxes[bb].begin();
						for (; it != tmp_bounding_boxes[bb].end(); it++)
						{
							if ((*it).exist)
							{
								bounding_boxes[i].push_back(*it);
								order.score = (*it).score;
								order.oriOrder = count;
								bounding_scores[i].push_back(order);
								count++;
							}
						}
					}

					//ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", 0);
					after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}

			}

			std::vector<ZQ_CNN_OrderScore> firstOrderScore;
			int count = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}


			//the first stage's nms
			if (count < 1) return false;
			double t15 = omp_get_wtime();
			ZQ_CNN_BBoxUtils::_nms(firstBbox, firstOrderScore, nms_thresh[0], "Union", 0, 1);
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(firstBbox, width, height,true);
			double t16 = omp_get_wtime();
			if (show_debug_info)
				printf("nms cost: %.3f ms\n", 1000 * (t16 - t15));
			if (show_debug_info)
				printf("first stage candidate count: %d\n", count);
			double t3 = omp_get_wtime();
			if (show_debug_info)
				printf("stage 1: cost %.3f ms\n", 1000 * (t3 - t1));
			return true;
		}

		bool _Rnet_stage(const ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& firstBbox, std::vector<ZQ_CNN_BBox>& secondBbox)
		{
			double t3 = omp_get_wtime();
			secondBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = firstBbox.begin();
			std::vector<ZQ_CNN_OrderScore> secondScore;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int r_count = 0;
			for (; it != firstBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						src_off_x.push_back(off_x);
						src_off_y.push_back(off_y);
						src_rect_w.push_back(rect_w);
						src_rect_h.push_back(rect_h);
						r_count++;
						secondBbox.push_back(*it);
					}
				}
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)r_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)r_count / batch_size);
				per_num = batch_size;
			}
			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_rnet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox> > task_secondBbox(need_thread_num);
			

			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(r_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_secondBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_secondBbox[i][j] = secondBbox[st_id + j];
					}
				}
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					if (task_src_off_x.size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_rnet_images[pp], rnet_size, rnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					rnet[0].Forward(task_rnet_images[pp]);
					const ZQ_CNN_Tensor4D* score = rnet[0].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = rnet[0].GetBlobByName("conv5-2");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int task_count = 0;
					for (int i = 0; i < task_secondBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[1])
						{
							for (int j = 0; j < 4; j++)
								task_secondBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							task_secondBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_secondBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_secondBbox[pp][i].exist = false;
						}
					}
					if (task_count < 1)
					{
						task_secondBbox[pp].clear();
						continue;
					}
					for (int i = task_secondBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_secondBbox[pp][i].exist)
							task_secondBbox[pp].erase(task_secondBbox[pp].begin() + i);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					if (task_src_off_x.size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_rnet_images[pp], rnet_size, rnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					rnet[thread_id].Forward(task_rnet_images[pp]);
					const ZQ_CNN_Tensor4D* score = rnet[thread_id].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = rnet[thread_id].GetBlobByName("conv5-2");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int task_count = 0;
					for (int i = 0; i < task_secondBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[1])
						{
							for (int j = 0; j < 4; j++)
								task_secondBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							task_secondBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_secondBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_secondBbox[pp][i].exist = false;
						}
					}
					if (task_count < 1)
					{
						task_secondBbox[pp].clear();
						continue;
					}
					for (int i = task_secondBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_secondBbox[pp][i].exist)
							task_secondBbox[pp].erase(task_secondBbox[pp].begin() + i);
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_secondBbox[i].size();
			}
			secondBbox.resize(count);
			secondScore.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_secondBbox[i].size(); j++)
				{
					secondBbox[id] = task_secondBbox[i][j];
					secondScore[id].score = secondBbox[id].score;
					secondScore[id].oriOrder = id;
					id++;
				}
			}

			//ZQ_CNN_BBoxUtils::_nms(secondBbox, secondScore, nms_thresh[1], "Union");
			ZQ_CNN_BBoxUtils::_nms(secondBbox, secondScore, nms_thresh[1], "Min");
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(secondBbox, width, height, true);
			count = secondBbox.size();

			double t4 = omp_get_wtime();
			if (show_debug_info)
				printf("run Rnet [%d] times, candidate after nms: %d \n", r_count, count);
			if (show_debug_info)
				printf("stage 2: cost %.3f ms\n", 1000 * (t4 - t3));

			return true;
		}

		bool _Onet_stage(const ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& secondBbox, std::vector<ZQ_CNN_BBox>& thirdBbox)
		{
			double t4 = omp_get_wtime();
			thirdBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = secondBbox.begin();
			std::vector<ZQ_CNN_OrderScore> thirdScore;
			std::vector<ZQ_CNN_BBox> early_accept_thirdBbox;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int o_count = 0;
			for (; it != secondBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						if (!do_landmark && it->score > early_accept_thresh)
						{
							early_accept_thirdBbox.push_back(*it);
						}
						else
						{
							src_off_x.push_back(off_x);
							src_off_y.push_back(off_y);
							src_rect_w.push_back(rect_w);
							src_rect_h.push_back(rect_h);
							o_count++;
							thirdBbox.push_back(*it);
						}
					}
				}
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)o_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)o_count / batch_size);
				per_num = batch_size;
			}

			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_onet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox> > task_thirdBbox(need_thread_num);

			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(o_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_thirdBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_thirdBbox[i][j] = thirdBbox[st_id + j];
					}
				}
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_onet_images[pp], onet_size, onet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					onet[0].Forward(task_onet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* score = onet[0].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = onet[0].GetBlobByName("conv6-2");
					const ZQ_CNN_Tensor4D* keyPoint = onet[0].GetBlobByName("conv6-3");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					const float* keyPoint_ptr = 0;
					if (keyPoint != 0)
						keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int keyPoint_sliceStep = 0;
					if (keyPoint != 0)
						keyPoint_sliceStep = keyPoint->GetSliceStep();
					int task_count = 0;
					ZQ_CNN_OrderScore order;
					for (int i = 0; i < task_thirdBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[2])
						{
							for (int j = 0; j < 4; j++)
								task_thirdBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							if (keyPoint != 0)
							{
								for (int num = 0; num < 5; num++)
								{
									task_thirdBbox[pp][i].ppoint[num] = task_thirdBbox[pp][i].col1 +
										(task_thirdBbox[pp][i].col2 - task_thirdBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
									task_thirdBbox[pp][i].ppoint[num + 5] = task_thirdBbox[pp][i].row1 +
										(task_thirdBbox[pp][i].row2 - task_thirdBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
								}
							}
							task_thirdBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_thirdBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_thirdBbox[pp][i].exist = false;
						}
					}

					if (task_count < 1)
					{
						task_thirdBbox[pp].clear();
						continue;
					}
					for (int i = task_thirdBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_thirdBbox[pp][i].exist)
							task_thirdBbox[pp].erase(task_thirdBbox[pp].begin() + i);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_onet_images[pp], onet_size, onet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					onet[thread_id].Forward(task_onet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* score = onet[thread_id].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = onet[thread_id].GetBlobByName("conv6-2");
					const ZQ_CNN_Tensor4D* keyPoint = onet[thread_id].GetBlobByName("conv6-3");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					const float* keyPoint_ptr = 0;
					if (keyPoint != 0)
						keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int keyPoint_sliceStep = 0;
					if (keyPoint != 0)
						keyPoint_sliceStep = keyPoint->GetSliceStep();
					int task_count = 0;
					ZQ_CNN_OrderScore order;
					for (int i = 0; i < task_thirdBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[2])
						{
							for (int j = 0; j < 4; j++)
								task_thirdBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							if (keyPoint != 0)
							{
								for (int num = 0; num < 5; num++)
								{
									task_thirdBbox[pp][i].ppoint[num] = task_thirdBbox[pp][i].col1 +
										(task_thirdBbox[pp][i].col2 - task_thirdBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
									task_thirdBbox[pp][i].ppoint[num + 5] = task_thirdBbox[pp][i].row1 +
										(task_thirdBbox[pp][i].row2 - task_thirdBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
								}
							}
							task_thirdBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_thirdBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_thirdBbox[pp][i].exist = false;
						}
					}

					if (task_count < 1)
					{
						task_thirdBbox[pp].clear();
						continue;
					}
					for (int i = task_thirdBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_thirdBbox[pp][i].exist)
							task_thirdBbox[pp].erase(task_thirdBbox[pp].begin() + i);
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_thirdBbox[i].size();
			}
			thirdBbox.resize(count);
			thirdScore.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_thirdBbox[i].size(); j++)
				{
					thirdBbox[id] = task_thirdBbox[i][j];
					thirdScore[id].score = task_thirdBbox[i][j].score;
					thirdScore[id].oriOrder = id;
					id++;
				}
			}

			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(thirdBbox, width, height, false);

			ZQ_CNN_OrderScore order;
			for (int i = 0; i < early_accept_thirdBbox.size(); i++)
			{
				order.score = early_accept_thirdBbox[i].score;
				order.oriOrder = count++;
				thirdScore.push_back(order);
				thirdBbox.push_back(early_accept_thirdBbox[i]);
			}
			
			ZQ_CNN_BBoxUtils::_nms(thirdBbox, thirdScore, nms_thresh[2], "Min");
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Onet [%d] times, candidate before nms: %d \n", o_count, count);
			if (show_debug_info)
				printf("stage 3: cost %.3f ms\n", 1000 * (t5 - t4));

			return true;
		}


		bool _Lnet_stage(const ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& thirdBbox, std::vector<ZQ_CNN_BBox>& fourthBbox)
		{
			double t4 = omp_get_wtime();
			fourthBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = thirdBbox.begin();
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int l_count = 0;
			for (; it != thirdBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						l_count++;
						fourthBbox.push_back(*it);
					}
				}
			}
			std::vector<ZQ_CNN_BBox> copy_fourthBbox = fourthBbox;
			ZQ_CNN_BBoxUtils::_square_bbox(copy_fourthBbox, width, height);
			for (it = copy_fourthBbox.begin(); it != copy_fourthBbox.end(); ++it)
			{
				int off_x = it->col1;
				int off_y = it->row1;
				int rect_w = it->col2 - off_x;
				int rect_h = it->row2 - off_y;
				src_off_x.push_back(off_x);
				src_off_y.push_back(off_y);
				src_rect_w.push_back(rect_w);
				src_rect_h.push_back(rect_h);
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)l_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)l_count / batch_size);
				per_num = batch_size;
			}

			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_lnet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox> > task_fourthBbox(need_thread_num);
			
			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(l_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_fourthBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_fourthBbox[i][j] = copy_fourthBbox[st_id + j];
					}
				}
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					if (task_src_off_x.size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_lnet_images[pp], lnet_size, lnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					lnet[0].Forward(task_lnet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* keyPoint = lnet[0].GetBlobByName("conv6-3");
					const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int keyPoint_sliceStep = keyPoint->GetSliceStep();
					for (int i = 0; i < task_fourthBbox[pp].size(); i++)
					{
						for (int num = 0; num < 5; num++)
						{
							task_fourthBbox[pp][i].ppoint[num] = task_fourthBbox[pp][i].col1 +
								(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
							task_fourthBbox[pp][i].ppoint[num + 5] = task_fourthBbox[pp][i].row1 +
								(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
						}
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					if (task_src_off_x.size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_lnet_images[pp], lnet_size, lnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					lnet[thread_id].Forward(task_lnet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* keyPoint = lnet[thread_id].GetBlobByName("conv6-3");
					const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int keyPoint_sliceStep = keyPoint->GetSliceStep();
					for (int i = 0; i < task_fourthBbox[pp].size(); i++)
					{
						for (int num = 0; num < 5; num++)
						{
							task_fourthBbox[pp][i].ppoint[num] = task_fourthBbox[pp][i].col1 +
								(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
							task_fourthBbox[pp][i].ppoint[num + 5] = task_fourthBbox[pp][i].row1 +
								(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
						}
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_fourthBbox[i].size();
			}
			fourthBbox.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_fourthBbox[i].size(); j++)
				{
					memcpy(fourthBbox[id].ppoint, task_fourthBbox[i][j].ppoint, sizeof(float) * 10);
					id++;
				}
			}
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Lnet [%d] times \n", l_count);
			if (show_debug_info)
				printf("stage 4: cost %.3f ms\n", 1000 * (t5 - t4));

			return true;

		}


		bool _Lnet106_stage(const ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, std::vector<ZQ_CNN_BBox>& thirdBbox, std::vector<ZQ_CNN_BBox106>& resultBbox)
		{
			double t4 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox> fourthBbox;
			std::vector<ZQ_CNN_BBox>::iterator it = thirdBbox.begin();
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int l_count = 0;
			for (; it != thirdBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						l_count++;
						fourthBbox.push_back(*it);
					}
				}
			}
			std::vector<ZQ_CNN_BBox> copy_fourthBbox = fourthBbox;
			ZQ_CNN_BBoxUtils::_square_bbox(copy_fourthBbox, width, height);
			for (it = copy_fourthBbox.begin(); it != copy_fourthBbox.end(); ++it)
			{
				int off_x = it->col1;
				int off_y = it->row1;
				int rect_w = it->col2 - off_x;
				int rect_h = it->row2 - off_y;
				src_off_x.push_back(off_x);
				src_off_y.push_back(off_y);
				src_rect_w.push_back(rect_w);
				src_rect_h.push_back(rect_h);
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)l_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)l_count / batch_size);
				per_num = batch_size;
			}

			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_lnet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox106> > task_fourthBbox(need_thread_num);

			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(l_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_fourthBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_fourthBbox[i][j].col1 = copy_fourthBbox[st_id + j].col1;
						task_fourthBbox[i][j].col2 = copy_fourthBbox[st_id + j].col2;
						task_fourthBbox[i][j].row1 = copy_fourthBbox[st_id + j].row1;
						task_fourthBbox[i][j].row2 = copy_fourthBbox[st_id + j].row2;
						task_fourthBbox[i][j].area = copy_fourthBbox[st_id + j].area;
						task_fourthBbox[i][j].score = copy_fourthBbox[st_id + j].score;
						task_fourthBbox[i][j].exist = copy_fourthBbox[st_id + j].exist;
					}
				}
			}

			resultBbox.resize(l_count);
			for (int i = 0; i < l_count; i++)
			{
				resultBbox[i].col1 = fourthBbox[i].col1;
				resultBbox[i].col2 = fourthBbox[i].col2;
				resultBbox[i].row1 = fourthBbox[i].row1;
				resultBbox[i].row2 = fourthBbox[i].row2;
				resultBbox[i].score = fourthBbox[i].score;
				resultBbox[i].exist = fourthBbox[i].exist;
				resultBbox[i].area = fourthBbox[i].area;
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					if (task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_lnet_images[pp], lnet_size, lnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					lnet[0].Forward(task_lnet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* keyPoint = lnet[0].GetBlobByName("conv6-3");
					const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int keypoint_num = keyPoint->GetC() / 2;
					int keyPoint_sliceStep = keyPoint->GetSliceStep();
					for (int i = 0; i < task_fourthBbox[pp].size(); i++)
					{
						for (int num = 0; num < keypoint_num; num++)
						{
							if ((num >= 33 && num < 43) || (num >= 64 && num < 72) || (num >= 84 && num < 104))
							{
								task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
									(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2]/**0.25*/;
								task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
									(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1]/**0.25*/;
							}
							else
							{
								task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
									(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2];
								task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
									(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1];
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					if (task_src_off_x.size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_lnet_images[pp], lnet_size, lnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
						continue;
					}
					double t31 = omp_get_wtime();
					lnet[thread_id].Forward(task_lnet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* keyPoint = lnet[thread_id].GetBlobByName("conv6-3");
					const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int keypoint_num = keyPoint->GetC() / 2;
					int keyPoint_sliceStep = keyPoint->GetSliceStep();
					for (int i = 0; i < task_fourthBbox[pp].size(); i++)
					{
						for (int num = 0; num < keypoint_num; num++)
						{
							if ((num >= 33 && num < 43) || (num >= 64 && num < 72) || (num >= 84 && num < 104))
							{
								task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
									(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2]/**0.5*/;
								task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
									(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1]/**0.5*/;
							}
							else
							{
								task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
									(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2];
								task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
									(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1];
							}
						}
					}
				}
			}
			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_fourthBbox[i].size();
			}
			resultBbox.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_fourthBbox[i].size(); j++)
				{
					memcpy(resultBbox[id].ppoint, task_fourthBbox[i][j].ppoint, sizeof(float) * 212);
					id++;
				}
			}
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Lnet [%d] times \n", l_count);
			if (show_debug_info)
				printf("stage 4: cost %.3f ms\n", 1000 * (t5 - t4));

			return true;
		}

		void _select(std::vector<ZQ_CNN_BBox>& bbox, int limit_num, int width, int height)
		{
			int in_num = bbox.size();
			if (limit_num >= in_num)
				return;
			bbox.resize(limit_num);
		}
	};
}
#endif
