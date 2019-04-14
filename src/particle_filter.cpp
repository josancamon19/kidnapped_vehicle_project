/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 100;  // TODO: Set the number of particles

    default_random_engine gen;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {

        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);

        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    default_random_engine gen;

    for (int i = 0; i < particles.size(); ++i) {

        Particle p = particles[i];

        double x, y, theta;
        if (fabs(yaw_rate) > 0.001) {

            double c = p.theta + (yaw_rate * delta_t);
            double vy = (velocity / yaw_rate);

            x = p.x + vy * (sin(c) - sin(p.theta));
            y = p.y + vy * (cos(p.theta) - cos(c));
            theta = p.theta + yaw_rate * delta_t;
        } else {
            x = p.x + velocity * cos(p.theta) * delta_t;
            y = p.y + velocity * sin(p.theta) * delta_t;
            theta = p.theta;
        }

        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    double c1 = 1.0 / (2 * M_PI * sig_x * sig_y);
    double cx = 2 * sig_x * sig_x;
    double cy = 2 * sig_y * sig_y;

    for (int i = 0; i < num_particles; ++i) {

        double mvgd = 1;

        for (int j = 0; j < observations.size(); ++j) {
            
            Particle p = particles[j];

            double transform_obs_x, transform_obs_y;

            double obs_x = observations[j].x;
            double obs_y = observations[j].y;

            transform_obs_x = p.x + cos(p.theta) * obs_x - sin(p.theta) * obs_y;
            transform_obs_y = p.y + sin(p.theta) * obs_x + cos(p.theta) * obs_y;

            vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
            vector<double> trans_obs_distances(landmarks.size());

            for (int k = 0; k < landmarks.size(); ++k) {

                double l_x = landmarks[k].x_f;
                double l_y = landmarks[k].y_f;

                double obs_distance = sqrt(pow(p.x - l_x, 2) + pow(p.y - l_y, 2));
                if (obs_distance <= sensor_range) {
                    trans_obs_distances[k] = sqrt(pow(transform_obs_x - l_x, 2) + pow(transform_obs_y - l_y, 2));
                } else {
                    trans_obs_distances[k] = 99999;
                }
            }

            int min_pos = distance(trans_obs_distances.begin(),
                                   min_element(trans_obs_distances.begin(), trans_obs_distances.end()));

            float nn_x = landmarks[min_pos].x_f;
            float nn_y = landmarks[min_pos].y_f;

            double x_diff = transform_obs_x - nn_x;
            double y_diff = transform_obs_y - nn_y;

            double exponential_part = ((x_diff * x_diff) / cx) + ((y_diff * y_diff) / cy);

            double result = c1 * exp(-exponential_part);
            mvgd *= result;
        }

        weights[i] = mvgd;
        particles[i].weight = mvgd;

    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    vector<Particle> new_particles(num_particles);

    default_random_engine gen;
    for (int i = 0; i < num_particles; ++i) {
        discrete_distribution<int> index(weights.begin(), weights.end());
        new_particles[i] = particles[index(gen)];
    }

    particles = new_particles;


}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}