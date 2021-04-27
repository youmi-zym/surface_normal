#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
#define DEBUG
#define DEEPDEBUG
#define COLORDEPTH
*/
using namespace cv;
using namespace std;
const float eps = 1e-8;

float fcxcy[3];//SET THE CAMERA PARAMETERS  F CX CY.
int  WINDOWSIZE=7;//SET SEARCH WINDOWSIZE(SUGGEST 15).
float Threshold=0.1;//SET THE threshold (SUGGEST 0.1-0.2).

void cvFitPlane(const Mat & points, float* plane);
void CallFitPlane(const Mat& depth,int * points,int i,int j,float *plane12);
void search_plane_neighbor(Mat &img, int i, int j, int* result);
bool convert_direction(float * sn,int i,int j,float d);
Mat calplanenormal(Mat  &src);

void search_plane_neighbor(Mat &img, int i, int j, int* result)
{
    int cols =img.cols;
    int rows =img.rows; 
    for (int ii=0; ii<WINDOWSIZE*WINDOWSIZE;ii++)
    {
       result[ii]=0;
    }
	float center_depth = img.at<float>(i,j);
    for (int idx=0; idx<WINDOWSIZE;idx++)
    {
	   for (int idy=0; idy<WINDOWSIZE;idy++)
       {
	       int rx= i-int(WINDOWSIZE/2)+idx;
	       int ry= j-int(WINDOWSIZE/2)+idy;
	       if(  rx>= rows || ry>=cols )
               continue;
           if( rx < 0 || ry < 0 )
               continue;
	       if( img.at<float>(rx,ry)==0.0)
               continue;
	       if( abs(img.at<float>(rx,ry)-center_depth)<=Threshold*center_depth )
               result[idx*WINDOWSIZE+idy]=1;
	   }
    }
}

bool convert_direction(float * sn,int i,int j,float d)
{
	float f =fcxcy[0];
	float cx=fcxcy[1];
	float cy=fcxcy[2];
	float x = (j - cx) *d * 1.0 / f;
    float y = (i - cy) *d * 1.0 / f;
    float z = d;
	// Vec3f world_center=Vec3f(0, 0, 0);
	Vec3f world_pos = Vec3f(x-0, y-0, z-0);
	Vec3f surface_normal = Vec3f(sn[0],sn[1],sn[2]);
	float cos_angle = world_pos.dot(surface_normal);
	if (cos_angle >= 0)
	    return false;
	else 
        return true;
 
}

// Ax+by+cz=D
void CallFitPlane(const Mat& depth,int * points,int i,int j,float *plane12) 
{
	float f =fcxcy[0];
	float cx=fcxcy[1];
	float cy=fcxcy[2];
	vector<float>X_vector;
	vector<float>Y_vector;
	vector<float>Z_vector;
	for(int num_point=0; num_point<WINDOWSIZE*WINDOWSIZE;num_point++ )
    {
		if (points[num_point]==1) //search 已经处理了边界,此处不需要再处理了
        {
		    int point_i,point_j;
            // row index in window
		    point_i=floor(num_point/WINDOWSIZE);
            // col index in window
		    point_j=num_point-(point_i*WINDOWSIZE);
            // row index in image
		    point_i+=i-int(WINDOWSIZE/2);
            // col index in image
            point_j+=j-int(WINDOWSIZE/2);

		    float x = (point_j - cx) * depth.at<float>(point_i, point_j) * 1.0 / f;
		    float y = (point_i - cy) * depth.at<float>(point_i, point_j) * 1.0 / f;
		    float z = depth.at<float>(point_i,point_j);
		    X_vector.push_back(x);
		    Y_vector.push_back(y);
		    Z_vector.push_back(z);
		}
    }

    Mat points_mat(X_vector.size(), 3, CV_32FC1);

	if(X_vector.size()<3)
    { 
        plane12[0]=-1;
        plane12[1]=-1;
        plane12[2]=-1;
        plane12[3]=-1;
        return;
    }
	for (int ii=0;ii < X_vector.size(); ++ii)
    {
        points_mat.at<float>(ii, 0) = X_vector[ii];
        points_mat.at<float>(ii, 1) = Y_vector[ii];
        points_mat.at<float>(ii, 2) = Z_vector[ii];
	}

#ifdef DEEPDEBUG
    cout << "Plane Fitting with: " << X_vector.size() << " points" << endl;
    cout << "Points are: " << endl << points_mat << endl;
#endif
    // plane fitting, least square error
	cvFitPlane(points_mat, plane12);

	if(convert_direction(plane12,i,j,depth.at<float>(i,j)))
    {
		plane12[0]=-plane12[0];
		plane12[1]=-plane12[1];
		plane12[2]=-plane12[2];
    }

	X_vector.clear();
	Y_vector.clear();
	Z_vector.clear();

    points_mat.release();
}

void cvFitPlane(const Mat & points, float* plane)
{
	// Estimate geometric centroid.
	int nrows = points.rows;
	int ncols = points.cols;
	int type  = points.type();

    Mat centroid(1, ncols, type);
	for (int c = 0; c<ncols; c++)
    {
        centroid.at<float>(0, c) = 0;
		for (int r = 0; r < nrows; r++)
		{
            centroid.at<float>(0, c) += points.at<float>(r, c);
        }
        centroid.at<float>(0, c) /= nrows;
	}

#ifdef DEBUG

    cout << "Plane Center: " << centroid  << endl;

#endif
	// Subtract geometric centroid from each point, get vector from plane center to points
    // please refer to https://www.notion.so/Plane-Fitting-09c8cbf14d2044b7b00614e7630798e5 for plane fitting
    Mat vPC(nrows, ncols, type, Scalar(0));
	for (int r = 0; r<nrows; r++)
		for (int c = 0; c<ncols; c++)
            vPC.at<float>(r, c) = points.at<float>(r, c) - centroid.at<float>(0, c);

	// Evaluate SVD of covariance matrix.
    Mat A = vPC.clone();
    Mat VT(ncols, ncols, type);

#ifdef DEEPDEBUG

    cout << "SVD, A[:3, :3]: " << endl << A.rowRange(0, 3).colRange(0, 3)  << endl;

#endif

    Mat w, u;
    SVD::compute(A, w, u, VT, cv::SVD::FULL_UV);

    Mat V = VT.clone().t();

#ifdef DEBUG

    cout << "SVD,  W: " << endl << w << endl;
    cout << "SVD, VT: " << endl << VT<< endl;

#endif

	// Assign plane coefficients by singular vector corresponding to smallest singular value.
    // V^T * N = (0, 0, 1)^T => N = V * (0, 0, 1)^T = (v1 v2 v3) * (0, 0, 1)^T = v3
	plane[ncols] = 0;
    for(int r=0; r < V.rows; r++)
    {
        plane[r] = V.at<float>(r, V.cols-1);
        plane[ncols] += plane[r] * centroid.at<float>(r);
    }

#ifdef DEBUG
    cout << "Surface Normal: " << endl << plane[0] << " " << plane[1] << " " << plane[2] << " " << plane[3] << endl;
#endif

	// Release allocated resources.
    centroid.release();
    vPC.release();
    A.release();
    VT.release();
    V.release();
    w.release();
    u.release();
}

Mat calplanenormal(Mat  &src)
{
     Mat normals = Mat::zeros(src.size(),CV_32FC3);
	 src.convertTo(src,CV_32FC1);
	 src*=1.0;
	 int cols =src.cols;
	 int rows =src.rows;

	 int * plane_points = new int[WINDOWSIZE*WINDOWSIZE];
	 float * plane12 = new float[4];
	 for (int i=0;i< rows;i++)
     {
		for (int j=0;j< cols;j++)
        {
            //for kitti and nyud test
            if(fabs(src.at<float>(i,j) < eps) || src.at<float>(i,j)==0.0)
                continue;
            //for:nyud train
            //  if(src.at<float>(i,j)<=4000.0)continue;   

#ifdef DEEPDEBUG
            cout << "src[" << i << "][" << j << "] =: " << src.at<float>(i,j) << endl;
#endif

			search_plane_neighbor(src, i, j, plane_points);

#ifdef DEEPDEBUG
            cout << "Window: " << endl;
            for(int i = 0; i< WINDOWSIZE*WINDOWSIZE; i++)
            {
                cout << plane_points[i];
                if(i % WINDOWSIZE == WINDOWSIZE-1)
                    cout << endl;
                else
                    cout << " ";

            }
            cout << endl;
#endif

			CallFitPlane(src,plane_points,i,j,plane12);
			Vec3f d = Vec3f(plane12[0],plane12[1],plane12[2]);
			Vec3f n = normalize(d);
			normals.at<Vec3f>(i, j) = n;
		}
    }

#ifdef DEBUG
    double mn, mx;
    minMaxLoc(normals, &mn, &mx);

    cout << "Normals in range: [" << mn << ", " << mx << "]" << endl;
#endif

    // normals are ranged in [-1, 1], convert to [0, 255]
	Mat res = Mat::zeros(src.size(),CV_32FC3);
    for (int i=0;i<rows;i++)
    {
       for (int j=0;j<cols;j++)
       {
           res.at<Vec3f>(i, j)[0] = (normals.at<Vec3f>(i, j)[0] + 1.0) * 255.0 / 2.0;
           res.at<Vec3f>(i, j)[1] = (normals.at<Vec3f>(i, j)[1] + 1.0) * 255.0 / 2.0;
           res.at<Vec3f>(i, j)[2] = (normals.at<Vec3f>(i, j)[2] + 1.0) * 255.0 / 2.0;
       }
    }
    res.convertTo(res, CV_8UC3);
    cvtColor(res, res, COLOR_BGR2RGB);

	delete[] plane12;
	delete[] plane_points;
	normals.release();

	return res;
}

int main(int argc, char** argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./main data_type path_to_depth path_to_save_depth scale_to_real_depth" << endl;
        return 1;
    }
    
    string data_type(argv[1]);
    string input_file(argv[2]);
    string output_file(argv[3]);
    float scale = atof(argv[4]);
    cout << "Depth after read will scale to 1.0/" << scale << endl;

    Mat src = imread(input_file, IMREAD_UNCHANGED);
    // Mat src = imread(input_file, IMREAD_GRAYSCALE);
    src.convertTo(src, CV_32FC1);
    src = src / scale;
    double mn, mx;
    minMaxLoc(src, &mn, &mx);
    cout << "Depth in range: [" << mn << ", " << mx << "]" << endl;

    float f, cx, cy;
    if(data_type == string("nyudv2"))
    {
        // set focal_length, cx and cy
        f = 578.0; cx = 318.9; cy = 242.7;
        fcxcy[0] = f; fcxcy[1] = cx; fcxcy[2] = cy;
        WINDOWSIZE=7;
        Threshold=0.1;
    }
    else if(data_type == string("kitti2015"))
    {
        f = 721.5377; cx = 609.5593; cy = 172.8540;
        fcxcy[0] = f; fcxcy[1] = cx; fcxcy[2] = cy;
        WINDOWSIZE=15;
        Threshold=0.1;
    }
    else if(data_type == string("kitti2012"))
    {
        f = 707.0912; cx = 601.8873; cy = 183.1104;
        fcxcy[0] = f; fcxcy[1] = cx; fcxcy[2] = cy;
        WINDOWSIZE=15;
        Threshold=0.1;
    }
    else if(data_type == string("scannet"))
    {
        f = 571.623718; cx = 319.500000; cy = 239.500000;
        fcxcy[0] = f; fcxcy[1] = cx; fcxcy[2] = cy;
        WINDOWSIZE=7;
        Threshold=0.1;
    }
    else
    {
        cout << "Data Type error, only support [nyudv2, kitti2015, kitti2012, scannet] now!" << endl;
        cout << "Please add your data type in ./surface_normal.cpp" << endl;
        return 1;
    }


#ifdef COLORDEPTH
    src = src / mx * 255;
    src.convertTo(src, CV_8UC1);
    applyColorMap(src, src, COLORMAP_JET);
    imshow("src", src);
    waitKey(0);
#endif

    Mat res = calplanenormal(src);
    imwrite(output_file, res);

    cout << "Done!" << endl;

    return 0;
}

