#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "lsst/cpputils/python.h"
#include <math.h>
#include <cmath>
#include <vector>
#include <stdio.h>

namespace py = pybind11;

namespace lsst {
namespace pipe {
namespace tasks {

py::array_t<double> fixGamut(py::array_t<double, py::array::c_style | py::array::forcecast> & Lab_points,
                             double xn,
                             double yn,
                             double zn) {
  py::buffer_info Lab_buffer = Lab_points.request();
  auto Lab_ptr = Lab_points.unchecked<2>();
  py::array_t<double> result(Lab_buffer.shape);
  py::buffer_info result_buffer = result.request();
  auto result_ptr = result.mutable_unchecked<2>();

  for (int pixel_number=0; pixel_number < Lab_buffer.shape[0]; pixel_number++){
    double L = Lab_ptr(pixel_number, 0);
    double a = Lab_ptr(pixel_number, 1);
    double b = Lab_ptr(pixel_number, 2);

    double lum_save = L;

    // calculate various constants
    double chroma_sq = a*a + b*b;
    double s_sq = chroma_sq / (chroma_sq + L*L);

    // Store the signless tangent of hue. Hue is a quntity we keep conserved
    // to preserve the 'color' of a pixel under consideration.
    double tanh = abs(b/a);

    double Fy = (L + 16) /116;
    double Y = yn * pow(Fy, 3);
    double X = xn * pow((a/500 + Fy), 3);
    double Z = zn * pow((-1*b/200 + Fy), 3);

    // calculate rgb
    double R = 3.2410*X + -1.5374*Y + -0.4986*Z;
    double G = -0.9692*X + 1.8760*Y + 0.0416*Z;
    double B = 0.0556*X + -0.2040*Y + 1.0570*Z;

    double Lp = L;
    double new_a, new_b;
    double sat_factor = 1;
    // double sat_factor = 0.95;
    double new_chroma = chroma_sq;
    // Find the maximum in gamut lum at a given saturation
    while (R > 0.98 || G > 0.98 || B > 0.98) {
      new_chroma -= 0.1;
      if (new_chroma <0){
        break;
      }
      // new_chroma = sat_factor*s_sq * (L*L)/(1 - s_sq);
      new_a = std::copysign(sqrt(new_chroma/(1+tanh*tanh)), a);
      new_b = std::copysign(new_a*tanh, b);
      
      double Fy = (L + 16) /116;
      double Y = yn * pow(Fy, 3);
      double X = xn * pow((new_a/500 + Fy), 3);
      double Z = zn * pow((-1*new_b/200 + Fy), 3);

      // calculate rgb
      R = 3.2410*X + -1.5374*Y + -0.4986*Z;
      G = -0.9692*X + 1.8760*Y + 0.0416*Z;
      B = 0.0556*X + -0.2040*Y + 1.0570*Z;
    }
    result_ptr(pixel_number, 0) = L;
    result_ptr(pixel_number, 1) = new_a;
    result_ptr(pixel_number, 2) = new_b;

  }
  return result;
}

void wrapFixGamut(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrap([](auto &mod) {
        mod.def("_fixGamut", &fixGamut,"");
    });
}

}
}
}
