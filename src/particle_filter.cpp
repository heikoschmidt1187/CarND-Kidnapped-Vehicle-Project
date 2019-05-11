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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 10;  // TODO: Set the number of particles

  // initialize random engine for determining position and heading
   // TODO: make generator class member
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // clear particle vector in case a re-init is triggered
  particles.clear();
  weights.clear();

  // create the desired particles
  for(int i = 0; i < num_particles; ++i) {

    // construct the particle
    Particle part;

    // set counter value as particle id
    part.id = i;

    // initialize weight to 1
    part.weight = 1.0;

    // get Gaussian distributed values for x, y and theta
    part.x = dist_x(gen);
    part.y = dist_y(gen);
    part.theta = dist_theta(gen);

    // add the new particle to the filter's particle list
    particles.push_back(part);
  }
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

   // init generator once with mean 0 and add value later to the real mean
   // TODO: make generator class member
   std::default_random_engine gen;
   std::normal_distribution<double> dist_x(0, std_pos[0]);
   std::normal_distribution<double> dist_y(0, std_pos[1]);
   std::normal_distribution<double> dist_theta(0, std_pos[2]);

   // update each particle
   for(auto& p : particles) {
     // avoid multiple calculations
     double vel_div_yawrate = velocity / yaw_rate;
     double theta_plus_deltaTheta = p.theta + yaw_rate * delta_t;

     // calculate the new position and yaw angle
     p.x = p.x + vel_div_yawrate * (sin(theta_plus_deltaTheta) - sin(p.theta));
     p.y = p.y + vel_div_yawrate * (cos(p.theta) - cos(theta_plus_deltaTheta));
     p.theta = theta_plus_deltaTheta;

     // add noise to the values
     p.x += dist_x(gen);
     p.y += dist_y(gen);
     p.theta += dist_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   // loop over all observations and find the nearest neighbor
   for(auto& o : observations) {
     // current min Euclidian distance
     double min = std::numeric_limits<double>::max();

     // loop over all predicted landmarks and check for distance
     for(const auto& p : predicted) {
       double currentDist = dist(o.x, o.y, p.x, p.y);

       // if current distance is smaller than current min, this neighbor is closer
       if(currentDist < min) {
         min = currentDist;
         o.id = p.id;
       }
     }
   }
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

   // pre-calculate the Gaussian norm
   double gaussian_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

   // update weights for each particle
   for(auto& p : particles) {

     // vector for holding a list of predicted landmarks withing the sensor range
     std::vector<LandmarkObs> predictedLandmarks;

     for(const auto& m : map_landmarks.landmark_list) {

       // check if Euclidian distance between sensor and particle
       if(dist(p.x, p.y, m.x_f, m.y_f) < sensor_range) {
         predictedLandmarks.push_back(LandmarkObs{m.id_i, m.x_f, m.y_f});
       }
     }

     // as the observarions are in the car coordinate system, the  need to be transformed
     // to map coordinate system for further processing
     std::vector<LandmarkObs> transformedObservations;
     for(const auto& o : observations) {
       double tx = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
       double ty = p.y + o.y * sin(p.theta) + o.y * cos(p.theta);

       transformedObservations.push_back(LandmarkObs{o.id, tx, ty});
     }

     // associate the observations with the nearest predicted landmark
     dataAssociation(predictedLandmarks, transformedObservations);

     // set the current particle weight to 1 to ensure correct multiplication later
     p.weight = 1.0;

     // loop over all observations and get the associated predicted landmark
     for(const auto& t : transformedObservations) {

       // find the associated observation
       auto associatedObs = std::find_if(predictedLandmarks.begin(),
                                predictedLandmarks.end(),
                                [&](const LandmarkObs& ob) -> bool {
                                  return (ob.id == t.id);
                                });

       // calculate and multiply to weight
       double exponent = pow(associatedObs->x - std_landmark[0], 2) / (2 * pow(std_landmark[0], 2))
              + pow(associatedObs->y - std_landmark[1], 2) / (2 * pow(std_landmark[1], 2));

       p.weight *= gaussian_norm * exp(-exponent);
     }
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
