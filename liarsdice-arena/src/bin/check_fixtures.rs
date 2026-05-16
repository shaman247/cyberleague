// Runs each recorded loss against each candidate bot and prints what the bot
// would have done plus whether that move flips the outcome to a win.

use liarsdice_arena::bot::{MyBot, MyBotV2, MyBotV3, MyBotV4, MyBotV5, MyBotV6};
use liarsdice_arena::fixtures;
use liarsdice_arena::game::Strategy;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let bots: Vec<Box<dyn Strategy>> = vec![
        Box::new(MyBot::new(StdRng::seed_from_u64(1))),
        Box::new(MyBotV2::new(StdRng::seed_from_u64(2))),
        Box::new(MyBotV3::new(StdRng::seed_from_u64(3))),
        Box::new(MyBotV4::new(StdRng::seed_from_u64(4))),
        Box::new(MyBotV5::new(StdRng::seed_from_u64(5))),
        Box::new(MyBotV6::new(StdRng::seed_from_u64(6))),
    ];

    for mut b in bots {
        println!("\n=== {} ===", b.name());
        fixtures::run_all(&mut *b);
    }
}
