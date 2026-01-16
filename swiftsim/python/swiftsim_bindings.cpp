// SwiftSim Python Bindings
// Zero-copy numpy array interface to SoA physics engine

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../core/swarm_physics_soa.hpp"

namespace py = pybind11;
using namespace swiftsim;

// ============================================================================
// NUMPY ARRAY WRAPPER (zero-copy view into SoA data)
// ============================================================================
template<typename T>
py::array_t<T> wrap_array(T* data, size_t size) {
    // Create numpy array that views existing memory (no copy)
    return py::array_t<T>(
        {size},                    // shape
        {sizeof(T)},               // strides
        data,                      // data pointer
        py::cast(nullptr)          // base object (memory managed by C++)
    );
}

// ============================================================================
// PYTHON MODULE
// ============================================================================
PYBIND11_MODULE(swiftsim, m) {
    m.doc() = "SwiftSim - High-performance drone swarm physics simulator";

    // ========================================================================
    // SwarmParams
    // ========================================================================
    py::class_<SwarmParams>(m, "SwarmParams")
        .def(py::init<>())
        .def_readwrite("mass", &SwarmParams::mass)
        .def_readwrite("max_thrust", &SwarmParams::max_thrust)
        .def_readwrite("max_torque", &SwarmParams::max_torque)
        .def_readwrite("motor_tc", &SwarmParams::motor_tc)
        .def_readwrite("Ixx", &SwarmParams::Ixx)
        .def_readwrite("Iyy", &SwarmParams::Iyy)
        .def_readwrite("Izz", &SwarmParams::Izz)
        .def("hover_throttle", &SwarmParams::hover_throttle);

    // ========================================================================
    // DroneSwarmSoA - Main data container
    // ========================================================================
    py::class_<DroneSwarmSoA>(m, "DroneSwarm")
        .def(py::init<size_t>(), py::arg("count"))
        .def("resize", &DroneSwarmSoA::resize)
        .def("reset", &DroneSwarmSoA::reset)
        .def_readonly("count", &DroneSwarmSoA::count)
        .def_readonly("capacity", &DroneSwarmSoA::capacity)

        // Position arrays (numpy views)
        .def_property_readonly("pos_x", [](DroneSwarmSoA& s) {
            return wrap_array(s.pos_x, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("pos_y", [](DroneSwarmSoA& s) {
            return wrap_array(s.pos_y, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("pos_z", [](DroneSwarmSoA& s) {
            return wrap_array(s.pos_z, s.count);
        }, py::return_value_policy::reference_internal)

        // Velocity arrays
        .def_property_readonly("vel_x", [](DroneSwarmSoA& s) {
            return wrap_array(s.vel_x, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("vel_y", [](DroneSwarmSoA& s) {
            return wrap_array(s.vel_y, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("vel_z", [](DroneSwarmSoA& s) {
            return wrap_array(s.vel_z, s.count);
        }, py::return_value_policy::reference_internal)

        // Acceleration arrays
        .def_property_readonly("acc_x", [](DroneSwarmSoA& s) {
            return wrap_array(s.acc_x, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("acc_y", [](DroneSwarmSoA& s) {
            return wrap_array(s.acc_y, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("acc_z", [](DroneSwarmSoA& s) {
            return wrap_array(s.acc_z, s.count);
        }, py::return_value_policy::reference_internal)

        // Orientation (quaternion)
        .def_property_readonly("quat_w", [](DroneSwarmSoA& s) {
            return wrap_array(s.quat_w, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("quat_x", [](DroneSwarmSoA& s) {
            return wrap_array(s.quat_x, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("quat_y", [](DroneSwarmSoA& s) {
            return wrap_array(s.quat_y, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("quat_z", [](DroneSwarmSoA& s) {
            return wrap_array(s.quat_z, s.count);
        }, py::return_value_policy::reference_internal)

        // Angular velocity
        .def_property_readonly("omega_x", [](DroneSwarmSoA& s) {
            return wrap_array(s.omega_x, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("omega_y", [](DroneSwarmSoA& s) {
            return wrap_array(s.omega_y, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("omega_z", [](DroneSwarmSoA& s) {
            return wrap_array(s.omega_z, s.count);
        }, py::return_value_policy::reference_internal)

        // Motor inputs
        .def_property_readonly("motor_0", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_0, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_1", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_1, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_2", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_2, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_3", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_3, s.count);
        }, py::return_value_policy::reference_internal)

        // Motor filtered outputs
        .def_property_readonly("motor_f0", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_f0, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_f1", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_f1, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_f2", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_f2, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("motor_f3", [](DroneSwarmSoA& s) {
            return wrap_array(s.motor_f3, s.count);
        }, py::return_value_policy::reference_internal)

        // Flags
        .def_property_readonly("is_grounded", [](DroneSwarmSoA& s) {
            return wrap_array(s.is_grounded, s.count);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("is_active", [](DroneSwarmSoA& s) {
            return wrap_array(s.is_active, s.count);
        }, py::return_value_policy::reference_internal)

        // Convenience: get all positions as (N, 3) array
        .def("get_positions", [](DroneSwarmSoA& s) {
            py::array_t<float> result({s.count, size_t(3)});
            auto r = result.mutable_unchecked<2>();
            for (size_t i = 0; i < s.count; ++i) {
                r(i, 0) = s.pos_x[i];
                r(i, 1) = s.pos_y[i];
                r(i, 2) = s.pos_z[i];
            }
            return result;
        })

        // Convenience: set all motor inputs at once
        .def("set_motors", [](DroneSwarmSoA& s, py::array_t<float> motors) {
            auto m = motors.unchecked<2>();
            if (m.shape(0) != (py::ssize_t)s.count || m.shape(1) != 4) {
                throw std::runtime_error("Motors array must be (count, 4)");
            }
            for (size_t i = 0; i < s.count; ++i) {
                s.motor_0[i] = m(i, 0);
                s.motor_1[i] = m(i, 1);
                s.motor_2[i] = m(i, 2);
                s.motor_3[i] = m(i, 3);
            }
        })

        // Convenience: set uniform throttle for all drones
        .def("set_throttle", [](DroneSwarmSoA& s, float throttle) {
            for (size_t i = 0; i < s.count; ++i) {
                s.motor_0[i] = s.motor_1[i] = s.motor_2[i] = s.motor_3[i] = throttle;
            }
        });

    // ========================================================================
    // SwarmPhysicsSoA - Physics engine
    // ========================================================================
    py::class_<SwarmPhysicsSoA>(m, "SwarmPhysics")
        .def(py::init<>())
        .def_readwrite("params", &SwarmPhysicsSoA::params)

        .def("step", &SwarmPhysicsSoA::step,
             py::arg("swarm"), py::arg("dt"),
             "Step physics for all drones (auto-selects SIMD if available)")

        .def("step_scalar", &SwarmPhysicsSoA::step_scalar,
             py::arg("swarm"), py::arg("dt"),
             "Step physics using scalar code")

#ifdef __AVX2__
        .def("step_avx", &SwarmPhysicsSoA::step_avx,
             py::arg("swarm"), py::arg("dt"),
             "Step physics using AVX2 SIMD")
#endif

#ifdef _OPENMP
        .def("step_parallel", &SwarmPhysicsSoA::step_parallel,
             py::arg("swarm"), py::arg("dt"), py::arg("num_threads") = 0,
             "Step physics using OpenMP parallelization")
#endif

        .def("set_hover_throttle", &SwarmPhysicsSoA::setHoverThrottle,
             py::arg("swarm"),
             "Set all drones to hover throttle");

    // ========================================================================
    // Module-level info
    // ========================================================================
    m.attr("__version__") = "0.1.0";

#ifdef __AVX2__
    m.attr("HAS_AVX2") = true;
#else
    m.attr("HAS_AVX2") = false;
#endif

#ifdef _OPENMP
    m.attr("HAS_OPENMP") = true;
#else
    m.attr("HAS_OPENMP") = false;
#endif
}
