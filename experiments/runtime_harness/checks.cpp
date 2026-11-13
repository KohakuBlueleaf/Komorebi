// The checks every scenario ends with, the job and grid helpers, and the runner.
#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "api/dataflow/dataflow_api.h"
#include "checks.hpp"
#include "komorebi/device/km_core.hpp"

namespace scen {

    namespace {

        // A command as the cores saw it, for the checks: its group and the slot it went
        // to.
        struct Submitted {
            uint32_t first, count, slot;
            bool exit;
            bool done;
        };

        std::map<uint32_t, Submitted> submitted(const Sim& s) {
            std::map<uint32_t, Submitted> m;
            for (size_t k = 0; k < s.jobs.size(); k++) {
                const JobResult& r = s.jobs[k];
                if (r.status != kNotSubmitted) {
                    const Job& j = s.cfg.jobs[k];
                    m[r.sequence] = {
                        j.first, j.count, r.slot, false, r.status == kDone
                    };
                }
            }
            if (s.exit_job.status != kNotSubmitted) {
                // EXIT is completed as a group of every core, whatever its group field
                // says.
                m[s.exit_job.sequence] = {
                    0, s.cfg.cores, 0, true, s.exit_job.status == kDone
                };
            }
            return m;
        }

        std::string json(const std::string& s) {
            std::string out = "\"";
            for (char ch : s) {
                if (ch == '"' || ch == '\\') {
                    out += '\\';
                }
                out += ch;
            }
            return out + "\"";
        }

        std::string json(const std::vector<std::string>& v) {
            std::string out = "[";
            for (size_t i = 0; i < v.size(); i++) {
                out += (i ? ", " : "") + json(v[i]);
            }
            return out + "]";
        }

    }  // namespace

    std::string fmt(const char* f, unsigned a, unsigned b, unsigned c) {
        char buf[200];
        std::snprintf(buf, sizeof buf, f, a, b, c);
        return buf;
    }

    Job map_job(uint32_t first, uint32_t count, uint32_t tiles) {
        Job j;
        j.handler = kHandlerMap;
        j.first = first;
        j.count = count;
        j.a[1] = tiles;
        j.map = true;
        return j;
    }

    Job nop_job(uint32_t first, uint32_t count) {
        Job j;
        j.handler = kHandlerNop;
        j.first = first;
        j.count = count;
        return j;
    }

    // Settings for the fault cases: short bounds, so that a stranded kernel shows fast.
    Config fault_config(uint32_t cores) {
        Config c;
        c.cores = cores;
        c.row = 2;
        c.wait_limit = 8;
        c.idle_limit = 64;
        c.deadline = 200000;
        c.timeout = 1500000;
        return c;
    }

    bool all_done(const Sim& s) {
        for (const JobResult& r : s.jobs) {
            if (r.status != kDone) {
                return false;
            }
        }
        return s.exit_job.status == kDone;
    }

    // What every case ends with: every thread returned, no stub contract broken, state
    // words consistent, and each acknowledged command completed by its whole group.
    void common(const Sim& s, Check& c) {
        c.expect(
            s.all_returned, "every thread of every core returned within the timeout"
        );
        c.expect(s.violations.empty(), "no stub contract or landing violation");
        if (all_done(s)) {
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                c.expect(
                    (s.state(core, km::kStEnds) & 0xFFFFFF) == 0x010101 * km::kEndExit,
                    fmt("core %u's reader, writer and compute left on the EXIT", core)
                );
            }
        }
        const auto recs = submitted(s);
        std::map<uint32_t, std::map<uint32_t, uint32_t>> takes, reports, acks;
        std::map<uint32_t, uint32_t> runs;
        std::map<uint32_t, std::vector<Event>> by_sequence;
        for (const Event& e : s.events) {
            if (e.kind != kEventStop) {
                by_sequence[e.a].push_back(e);
            }
            if (e.kind == kEventTake && e.a != 0) {
                takes[e.a][e.core]++;
                runs[e.core]++;
            } else if (e.kind == kEventReport) {
                reports[e.a][e.core]++;
            } else if (e.kind == kEventAck) {
                acks[e.a][e.core]++;
                const auto it = recs.find(e.a);
                c.expect(
                    it != recs.end() && it->second.slot == e.b,
                    fmt("core %u acknowledged %u in ack word %u, the slot it was "
                        "submitted to",
                        e.core,
                        e.a,
                        e.b)
                );
            }
        }
        for (uint32_t core = 0; core < s.cfg.cores; core++) {
            c.expect(
                s.state(core, km::kStMagic) == (km::kStateMagic | 0x5A5Au << 8 | core),
                fmt("core %u start-up word intact", core)
            );
            c.expect(
                s.state(core, km::kStRuns) == runs[core],
                fmt(
                    "core %u counts %u commands taken, the log shows %u",
                    core,
                    s.state(core, km::kStRuns),
                    runs[core]
                )
            );
        }
        for (const auto& [sequence, r] : recs) {
            if (!r.done) {
                continue;
            }
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                const bool included = core - r.first < r.count;
                const uint32_t t = takes[sequence][core];
                c.expect(
                    t == (included ? 1u : 0u),
                    fmt("command %u taken %u times by core %u", sequence, t, core)
                );
                if (included) {
                    const uint32_t want = core == r.first ? 0 : 1;
                    c.expect(
                        reports[sequence][core] == want,
                        fmt(
                            "command %u: core %u sent %u reports",
                            sequence,
                            core,
                            reports[sequence][core]
                        )
                    );
                    c.expect(
                        acks[sequence][core] == (core == r.first ? 1u : 0u),
                        fmt(
                            "command %u: core %u sent %u acks",
                            sequence,
                            core,
                            acks[sequence][core]
                        )
                    );
                } else {
                    c.expect(
                        reports[sequence][core] + acks[sequence][core] == 0,
                        fmt(
                            "command %u: core %u, not in the group, reported",
                            sequence,
                            core
                        )
                    );
                }
            }
            // A core reports (or acks) only after every report to it for the record
            // landed, and it waits for exactly as many as its children in the tree.
            km_command_t rec = {};
            rec.group = km::make_group(r.first, r.count);
            std::map<uint32_t, std::vector<uint64_t>> landed;
            for (const Event& e : by_sequence[sequence]) {
                if (e.kind == kEventReportLand) {
                    landed[e.b].push_back(e.t);
                }
            }
            for (const Event& e : by_sequence[sequence]) {
                if (e.kind != kEventReport && e.kind != kEventAck) {
                    continue;
                }
                const uint32_t want = km::report_to(rec, e.core, s.cfg.tree_row).want;
                uint32_t before = 0;
                for (uint64_t t : landed[e.core]) {
                    before += t <= e.t;
                }
                c.expect(
                    before == want,
                    fmt("command %u: core %u sent its report after %u of the "
                        "reports to it landed",
                        sequence,
                        e.core,
                        before) +
                        fmt(", not %u", want)
                );
            }
        }
        for (size_t k = 0; k < s.jobs.size(); k++) {
            c.expect(
                s.jobs[k].data_bad == 0,
                fmt("job %u: %u output elements wrong", k, s.jobs[k].data_bad)
            );
        }
        if (!s.timed_out && s.all_returned) {
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
                    c.expect(
                        s.word(core, KM_CONTROL_DONE + j * 16) == 0,
                        fmt(
                            "core %u: completion counter %u is zero at the end", core, j
                        )
                    );
                }
            }
        }
    }

    // A tile that gave up acknowledges nothing afterwards: no report or ack from `core`
    // for command `sequence` or any later one.
    void silent_from(const Sim& s, Check& c, uint32_t core, uint32_t sequence) {
        for (const Event& e : s.events) {
            if (
                (e.kind == kEventReport || e.kind == kEventAck) && e.core == core &&
                km::newer(e.a + 1, sequence)
            ) {
                c.expect(
                    false,
                    fmt(
                        "core %u reported command %u after giving up on %u",
                        core,
                        e.a,
                        sequence
                    )
                );
            }
        }
    }

    void finish(const Sim& s, Outcome& o) {
        o.cycles = s.end_cycles;
        o.host_ms = s.host_ms;
        o.stuck = s.stuck;
        o.violations = s.violations;
        o.jobs = s.jobs.size();
        o.mailbox_peak = s.mailbox_peak;
        for (const JobResult& r : s.jobs) {
            o.done += r.status == kDone;
        }
    }

    void items_none() {
    }

    int run_cases(const std::vector<Case>& cases, int argc, char** argv) {
        if (argc != 2) {
            std::fprintf(stderr, "usage: scenarios <name> | list\n");
            return 2;
        }
        for (const Case& c : cases) {
            if (std::strcmp(argv[1], "list") == 0) {
                std::printf("%s\n", c.name);
            } else if (std::strcmp(argv[1], c.name) == 0) {
                Outcome o;
                c.fn(o);
                std::printf(
                    "{\"case\": %s, \"ok\": %s, \"fails\": %s, \"stuck\": %s, "
                    "\"violations\": %s, \"cycles\": %llu, \"host_ms\": %.1f, "
                    "\"jobs\": %zu, \"done\": %zu, \"mailbox_peak\": %u}\n",
                    json(c.name).c_str(),
                    o.check.fails.empty() ? "true" : "false",
                    json(o.check.fails).c_str(),
                    json(o.stuck).c_str(),
                    json(o.violations).c_str(),
                    static_cast<unsigned long long>(o.cycles),
                    o.host_ms,
                    o.jobs,
                    o.done,
                    o.mailbox_peak
                );
                return 0;
            }
        }
        return std::strcmp(argv[1], "list") == 0 ? 0 : 2;
    }

}  // namespace scen
