// for visualising results with the python turtle
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::time::Duration;
use std::fs::File;
use std::io::{BufReader, BufRead};

use bwinf_2::run::run;
use bwinf_2::necessaries::{In, Mode, save_path};

fn main() {
    let file = File::open("config.txt")
        .expect("can not open config file");

    let mut configs = BufReader::new(file).lines()
        .filter_map(|l| l.ok());

    let py_enabled = configs.nth(0).expect("Error parsing config.txt in line 1").split("=").nth(1).expect("Error parsing config.txt in line 1") == "true";

    let mode = match configs.nth(0).expect("Error parsing config.txt in line 2").split("=").nth(1).expect("Error parsing config.txt in line 2") {
        "normal" => Mode::Normal,
        "expensive" => Mode::Expensive,
        "cached" => Mode::Cached,
        "optimal" => Mode::Optimal,
        _ => panic!("Error parsing config.txt in line 2")
    };

    let custom_input_mode = match configs.nth(0).expect("Error parsing config.txt in line 3").split("=").nth(1).expect("Error parsing config.txt in line 3") {
        "normal" => Mode::Normal,
        "expensive" => Mode::Expensive,
        "cached" => Mode::Cached,
        "optimal" => Mode::Optimal,
        _ => panic!("Error parsing config.txt in line 3")
    };

    let inputs = configs.into_iter().collect::<Vec<String>>();

    if py_enabled {
        pyo3::prepare_freethreaded_python();
    
        Python::with_gil(|py| {
            // load functions from python files
            let show: Py<PyAny> = PyModule::from_code(py, include_str!("show.py"), "", "").unwrap()
                .getattr("show").unwrap().into();
            let get_new: Py<PyAny> = PyModule::from_code(py, include_str!("show.py"), "", "").unwrap()
                .getattr("get_new").unwrap().into();
    
            // for every preset file
            for i in inputs {
                let res = run(In::AFile(format!("inputs/{}.txt", i)), mode);
                if res.1.is_empty() {
                    println!("Coulnd't find a result")
                } else {
                    println!("Finished inputs/{}.txt in {:?} flying {:?}km", i, res.2, res.0);
                    // can be used to save the result in a file
                    // save_path("inputs/".to_string() + i + "_res.txt", res.1.clone());
                    show.call1(py, PyTuple::new(py, res.1)).unwrap();
                }
            }
    
            // for custom examples
            println!("Try drawing something yourself now!");
    
            while let Ok(a) = get_new.call0(py).unwrap().extract::<Vec<(f32, f32)>>(py) {
                if a.is_empty() {break;}
    
                println!("\n{} mode: ", Mode::to_string(custom_input_mode));
                eval_res(py, run(In::AVec(a.clone()), custom_input_mode), show.clone(), a.len());
            }
        })
    } else {
        for i in inputs {
            let res = run(In::AFile(format!("inputs/{}.txt", i)), mode);
            if res.1.is_empty() {
                println!("Coulnd't find a result")
            } else {
                println!("Finished inputs/{}.txt in {:?} flying {:?}km", i, res.2, res.0);
                save_path("inputs/".to_string() + &i + "_res.txt", res.1.clone());
            }
        }
    }
}

fn eval_res(py: Python, res: (f32, Vec<(f32, f32)>, Duration), show: Py<PyAny>, length: usize) {
    if res.1.is_empty() {
        println!("Coulnd't find a result ({:?} points, {:?})", length, res.2)
    } else {
        println!("Finished {:?} points in {:?} flying {:?}km", length, res.2, res.0);
        // println!("{:?}", res.1);
        show.call1(py, PyTuple::new(py, res.1)).unwrap();
    }
}


#[cfg(test)]
mod tests {
    use bwinf_2::necessaries::*;
    // Zum Testen der Funktion is_valid
    #[test]
    fn test1() {
        let list = vec![((1., 0.), (-1., 0.), (-10., 0.)), ((1., 7.), (10., 8.), (12., 0.)), ((0., -10.), (0., 0.), (-12., 0.)), ((1., 7.), (1., 5.), (-12., 5.)), ((2., 7.), (1., 5.), (-12., 5.)), ((0., 0.), (1., 1.), (1., 2.))];
        for (a, b, c) in list {
            assert_eq!(is_valid(Dot { x: a.0, y: a.1 }, Dot { x: b.0, y: b.1 }, Dot { x: c.0, y: c.1 }), true);
        }

        let n_list = vec![((-1., 0.), (0., -1.), (0., 0.)), ((1., 0.), (0., 1.), (0., 0.)), ((0., 1.), (0., -1.), (0., 0.)), ((0., -1.), (-1., 0.), (0., 0.))];
        for (a, b, c) in n_list {
            assert_eq!(is_valid(Dot { x: a.0, y: a.1 }, Dot { x: b.0, y: b.1 }, Dot { x: c.0, y: c.1 }), false);
        }
        assert_eq!(Dot {x: 0., y: 0.}.distance(&Dot {x: 1., y: 1.}), (2. as f32).sqrt())
    }
}