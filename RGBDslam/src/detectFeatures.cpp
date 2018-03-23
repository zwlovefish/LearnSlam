#include<iostream>
#include<slamBase.h>
using namespace std;

#include<opencv2/features2d/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
using namespace cv;
int main(int argc,char **argv)
{
  Mat rgb1=imread("./data/rgb1.png");
  Mat rgb2=imread("./data/rgb2.png");
  Mat depth1=imread("./data/depth1.png",-1);
  Mat depth2=imread("./data/depth2.png",-1);
  
  Ptr<FeatureDetector> detector;
  Ptr<DescriptorExtractor> descriptor;
  
  detector=ORB::create();
  descriptor=ORB::create();
  
  vector<KeyPoint> kp1,kp2;
  
  detector->detect(rgb1,kp1);
  detector->detect(rgb2,kp2);
  cout<<"key points of two images"<<kp1.size()<<","<<kp2.size()<<endl;
  //可视化，显示关键点
  Mat imgShow;
  drawKeypoints(rgb1,kp1,imgShow,cv::Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  imshow("keypoints",imgShow);
  imwrite("./data/keypoints.png",imgShow);
  waitKey(0);
  
  //计算描述子
  Mat desp1,desp2;
  descriptor->compute(rgb1,kp1,desp1);
  descriptor->compute(rgb2,kp2,desp2);
  //匹配描述子
  vector<DMatch> matches;
  BFMatcher matcher;
  matcher.match(desp1,desp2,matches);
  cout<<"find total"<<matches.size()<<".matches"<<endl;
  
  //可视化，显示匹配的特征
  Mat imgMatches;
  drawMatches(rgb1,kp1,rgb2,kp2,matches,imgMatches);
  imshow("matches",imgMatches);
  imwrite("./data/matches.png",imgMatches);
  waitKey(0);
  
  //筛选匹配，把距离太大的去掉
  //这里使用的准则是去掉大于四倍最小距离的匹配
  vector<DMatch> goodMatches;
  double minDis=9999;
  for(size_t i=0;i<matches.size();i++)
  {
    if(matches[i].distance<minDis)
      minDis=matches[i].distance;
  }
  cout<<"min dis ="<<minDis<<endl;
  for(size_t i=0;i<matches.size();i++)
  {
    if(matches[i].distance<10*minDis)
      goodMatches.push_back(matches[i]);
  }
  
  //显示good matches
  cout<<"good matches = "<<goodMatches.size()<<endl;
  drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,imgMatches);
  imshow("good matches",imgMatches);
  imwrite("./data/good_matches.png",imgMatches);
  waitKey(0);
  
  //计算图像间的运动关系
  //关键函数cv::solvePnPRansac()
  //为调用此函数准备必要的参数
  //第一个帧的三维点
  vector<Point3f> pts_obj;
  //第二个帧的图像点
  vector<Point2f> pts_img;
  //相机内参
  CAMERA_INTRINSIC_PARAMETERS C;
    C.cx=325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;
    
  for(size_t i=0;i<goodMatches.size();i++)
  {
    //query是第一个，train是第二个
    Point2f p=kp1[goodMatches[i].queryIdx].pt;
    ushort d=depth1.ptr<ushort>(int(p.y))[int(p.x)];
    if(d==0)
      continue;
    pts_img.push_back(Point2f(kp2[goodMatches[i].trainIdx].pt));
    //将（u,v,d）转为（x,y,z）
    Point3f pt(p.x,p.y,d);
    Point3f pd=point2dTo3d(pt,C);
    pts_obj.push_back(pd);
  }
  double CAMERA_MATRIX_DATA[3][3]={
    {C.fx,0,C.cx},
    {0,C.fy,C.cy},
    {0,0,1}
  };
  //构建相机矩阵
  Mat cameraMatrix(3,3,CV_64F,CAMERA_MATRIX_DATA);
  Mat rvec,tvec,inliers;
  cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,Mat(),rvec,tvec,false,100,1.0,0.95,inliers);
  cout<<"inliers:"<<inliers.rows<<endl;
  cout<<"R="<<rvec<<endl;
  cout<<"t="<<tvec<<endl;
  
  //画出inliers匹配
  vector<DMatch> matchesShow;
  for(size_t i=0;i<inliers.rows;i++)
  {
    matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
  }
  drawMatches(rgb1,kp1,rgb2,kp2,matchesShow,imgMatches);
  imshow("inlier marches",imgMatches);
  imwrite("./data/inliers.png",imgMatches);
  waitKey(0);
  
  return 0;
}