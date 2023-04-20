use std::fs::File;
use std::io::{BufReader, BufRead};
use std::thread;
use std::time::{Instant, Duration};

use crate::necessaries::*;

pub fn run(input: In, mode: Mode) -> (f32, Vec<(f32, f32)>, Duration) {
    let time = Instant::now();
    // setting this to false results in more efficient Caching, 
    // but turns off multithreading
    let multithreaded_cache = false;
    // its possible to pass a Vec<(f32, f32)> and a path
    let points: Vec<Dot> = match input {
        In::AVec(the_points) => the_points.into_iter()
            .map(|(x,y)| Dot { x, y })
            .collect(),
        In::AFile(ref filename) => {
            let file = File::open(filename)
                .expect(format!("can not open file {filename}").as_str());

            BufReader::new(file).lines()
                .filter_map(|l| l.ok())
                .map(parse_line)
                .collect()
        },
    };

    if mode == Mode::Cached && !multithreaded_cache {
        let res = start_cached(&points);
        return (res.0, Vec::from_iter(res.1.iter().map(|a| a.get_infos())), time.elapsed())
    }

    let mut results = Vec::new();
    for i in points.clone() {
        let p = points.clone();
        results.push(thread::spawn(move || i.start(&p, mode)))
    }

    let mut best: (f32, Vec<Dot>) = (f32::MAX, Vec::new());

    for i in results {
        let res = i.join().unwrap().unwrap();
        if res.0 < best.0 {
            best = res;
            // delete this if you want shorter paths, let it be for faster results
            if mode == Mode::Optimal {
                return (
                    best.0,
                    Vec::from_iter(best.1.iter().map(|a| a.get_infos())), time.elapsed()
                )
            }
        }
    }

    if best.1.is_empty() {
        return (0., Vec::new(), time.elapsed());
    }

    return (best.0, Vec::from_iter(best.1.iter().map(|a| a.get_infos())), time.elapsed())
}

fn parse_line(line: String) -> Dot {
    let line = line
        .split(" ")
        .take(2)
        .map(|v| v.parse::<f32>().unwrap())
        .collect::<Vec<f32>>();
    Dot { x: line[0], y: line[1] }
}