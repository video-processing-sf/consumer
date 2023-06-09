#pragma once
#include <opencv2/opencv.hpp>



namespace consumer
{

class ImageProcessor
{
public:
    ImageProcessor(std::shared_ptr<SharedMemory> shm) : sharedMemory_{shm}, processStream_{true}
    {
        framesStream_ = std::make_shared<cv::VideoCapture>(shm->GetBuffer()->data);
    }

    ~ImageProcessor()
    {
        cv::destroyAllWindows();
    }

    void ProcessStream()
    {
        cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
        cv::namedWindow("Motion Trajectory", cv::WINDOW_NORMAL);

        // Set the frame skip factor (process every 5th frame)
        const int FRAME_SKIP = 5;
        int frameCount = 0;

        std::vector<cv::Point2f> prevPoints, nextPoints;
        std::vector<uchar> status;
        std::vector<float> errors;

        cv::Mat prevFrame;

        cv::Mat trajectory(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

        while (processStream_) {
            cv::Mat frame;
            {
                auto buff = sharedMemory_->GetBuffer();
                std::lock_guard<std::shared_mutex> lock(buff->mutex);
                frame = cv::Mat(HEIGHT, WIDTH, CV_8UC1, buff->data);
            }

            frameCount++;

            cv::imshow("Camera Feed", frame);

            if (cv::waitKey(1) == 27) {
                break;
            }

            if (frameCount % FRAME_SKIP != 0) {
                continue;
            }

            if (prevPoints.empty()) {
                cv::goodFeaturesToTrack(frame, prevPoints, 500, 0.01, 10);
            } else {
                cv::Mat grayFrame, grayPrevFrame;
                cv::cvtColor(frame, grayFrame, cv::COLOR_GRAY2BGR);
                cv::cvtColor(prevFrame, grayPrevFrame, cv::COLOR_GRAY2BGR);

                cv::calcOpticalFlowPyrLK(grayPrevFrame, grayFrame, prevPoints, nextPoints, status, errors);

                for (size_t i = 0; i < prevPoints.size(); i++) {
                    if (status[i]) {
                        cv::Point2f p1 = prevPoints[i];
                        cv::Point2f p2 = nextPoints[i];
                        cv::arrowedLine(frame, p1, p2, cv::Scalar(0, 255, 0), 2);
                        cv::arrowedLine(trajectory, p1, p2, cv::Scalar(0, 255, 0), 2);
                    }
                }

                prevPoints = nextPoints;
            }

            frame.copyTo(prevFrame);

            cv::imshow("Motion Trajectory 3D", trajectory);
        }
    }


    void MakeVisualOdometry(const cv::Mat& K) {
        std::vector<cv::Point2f> pts1, pts2;
        std::vector<cv::Point3f> points3d;
        cv::Mat r_init = cv::Mat::eye(3, 3, CV_64FC1);
        cv::Mat t_init = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat rotation = r_init.clone();
        cv::Mat trans = t_init.clone();
        std::vector<double> x, z;
        
        cv::Mat frame;
        (*frameStream_) >> frame;
        cv::goodFeaturesToTrack(frame, pts1, 400, 0.01, 8, cv::Mat(), 21, false, 0.04);
        
        while (processStream_) {
            pts2 = trackFeatures(frame, pts1);
            
            cv::Mat E, mask;
            E = cv::findEssentialMat(pts2, pts1, K, cv::RANSAC, 0.999, 0.4, mask);
            
            std::vector<cv::Point2f> p1m, p2m;
            for (size_t j = 0; j < mask.rows; ++j) {
                if (mask.at<uchar>(j)) {
                    p1m.push_back(pts1[j]);
                    p2m.push_back(pts2[j]);
                }
            }
            
            cv::Mat R, t;
            cv::recoverPose(E, p1m, p2m, K, R, t);
            
            cv::Mat n_cloud = triangulation(R, t, p1m, p2m, K);
            cv::Mat points3d_t = n_cloud.rowRange(0, 3).t();
            
            double rep_error = cv::norm(n_cloud.row(2)) / n_cloud.cols;
            if (rep_error <= 8.0) {
                rotation = rotation * R;
                cv::Mat t1 = t / cv::norm(t);
                trans = trans - rotation * t1;
                x.push_back(trans.at<double>(0));
                z.push_back(-trans.at<double>(2));
            }
            
            pts1 = pts2;
            points3d = std::vector<cv::Point3f>(points3d_t.begin<cv::Point3f>(), points3d_t.end<cv::Point3f>());
            
            cv::Mat trajectory = cv::Mat::zeros(800, 800, CV_8UC3);
            double scale = 10.0;
            for (size_t i = 1; i < x.size(); ++i) {
                cv::Point p1(scale * x[i - 1] + 400, scale * z[i - 1] + 400);
                cv::Point p2(scale * x[i] + 400, scale * z[i] + 400);
                cv::line(trajectory, p1, p2, cv::Scalar(0, 255, 0), 1);
            }
            
            cv::imshow("Motion Trajectory 2D", trajectory);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }
    


protected:
    std::vector<cv::Point2f> trackFeatures(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& corners)
    {
        std::vector<cv::Point2f> p1, p2;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03);
        
        cv::calcOpticalFlowPyrLK(img1, img2, corners, p2, status, err, cv::Size(23, 23), 2, criteria);
        
        size_t i = 0;
        while (i < status.size()) {
            if (status[i] == 1) {
                p1.push_back(corners[i]);
                p2.push_back(p2[i]);
            }
            i++;
        }
        
        return p1;
    }

    cv::Mat triangulation(const cv::Mat& R, const cv::Mat& t, const std::vector<cv::Point2f>& pt1,
        const std::vector<cv::Point2f>& pt2, const cv::Mat& K) {
        cv::Mat cloud;
        cv::Mat ch1(pt1), ch2(pt2);
        cv::Mat pr = cv::Mat::eye(3, 4, CV_64FC1);
        cv::Mat pr_mat = K * pr;
        cv::Mat P = cv::Mat::zeros(3, 4, CV_64FC1);
        cv::hconcat(R, t, P);
        cv::Mat P1 = K * P;
        
        ch1 = ch1.t();
        ch2 = ch2.t();
        
        cv::triangulatePoints(pr_mat, P1, ch1, ch2, cloud);
        cloud = cloud.rowRange(0, 3) / cloud.row(3);
        
        return cloud;
    }


private:
    std::shared_ptr<SharedMemory> sharedMemory_;
    std::shared_ptr<cv::VideoCapture> framesStream_;
    std::atomic<bool> processStream_;
};

}

