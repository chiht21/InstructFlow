from __future__ import annotations

import importlib
import logging
import math
import os
import pathlib
import random
import time
import imageio
import traceback
import base64
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Union
import re

import numpy as np

from vtamp.environments.utils import Action, Environment, State
from vtamp.policies.utils import (
    ContinuousSampler,
    DiscreteSampler,
    Policy,
    Sampler,
    parse_code,
    query_llm,
)
from vtamp.utils import (
    are_files_identical,
    get_log_dir,
    get_previous_log_folder,
    parse_text_prompt,
    read_file,
    save_log,
    write_prompt,
)

_, _ = Action(), State()
log = logging.getLogger(__name__)


FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_domain"

def extract_subgoals_text_only(text):
    subgoals = []
    # 匹配形式 "subgoal": "xxx"
    pattern = r'"subgoal"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, text, re.IGNORECASE)

    for match in matches:
        if match.strip():
            subgoals.append(match.strip())

    return subgoals

def rejection_sample_csp(
    env: Environment,
    initial_state: State,
    plan_gen: Callable[[List[Union[int, float]]], List[Action]],
    domains_gen: List[Sampler],
    max_attempts: int = 10000,
) -> Union[List[Action], str]:
    """A constraint satisfaction strategy that randomly samples input vectors
    until it finds one that satisfies the constraints.

    If none are found, it returns the most common mode of failure.
    """
    violation_modes = Counter()
    for i in range(max_attempts):
        log.info(f"CSP Sampling iter {i}")
        domains = domains_gen(initial_state)
        input_vec = {name: domain.sample() for name, domain in domains.items()}
        _ = env.reset()
        ground_plan = plan_gen(initial_state, **input_vec)
        constraint_violated = False
        for ai, action in enumerate(ground_plan):
            _, _, _, info = env.step(action)
            if len(info["constraint_violations"]) > 0:
                violation_str = [
                    "Step {}, Action {}, Violation: {}".format(
                        ai, action.name, violation
                    )
                    for violation in info["constraint_violations"]
                ]
                violation_modes.update(violation_str)
                constraint_violated = True
                log.info(f"Constraint violation " + str(info["constraint_violations"]))
                break
        if not constraint_violated:
            return ground_plan, None, i

    return None, violation_modes, i


def import_constants_from_class(cls):
    # Get the module name from the class
    module_name = cls.__module__

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Import all uppercase attributes (assuming these are constants)
    for attribute_name in module.__all__:
        # Importing the attribute into the global namespace
        globals()[attribute_name] = getattr(module, attribute_name)
        print(f"Imported {attribute_name}: {globals()[attribute_name]}")


class Ours(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        max_csp_samples=10000,
        seed=0,
        use_cache=False,
        gaussian_blur=False,
        **kwargs,
    ):
        self.twin = twin
        self.use_cache = use_cache
        self.gaussian_blur = gaussian_blur
        self.seed = seed
        self.max_csp_samples = max_csp_samples
        self.max_feedbacks = max_feedbacks

        # Get environment specific prompt
        import_constants_from_class(twin.__class__) 
        prompt_fn = "prompt_{}".format(twin.__class__.__name__) + "2"
        prompt_fn2 = "prompt_{}".format(twin.__class__.__name__) + "3"
        prompt_fn3 = "prompt_{}".format(twin.__class__.__name__) + "4"
        prompt_fn4 = "prompt_{}".format(twin.__class__.__name__) + "6"
        prompt_fn5 = "prompt_{}".format(twin.__class__.__name__) + "5"

        prompt_path = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn)
        )

        prompt_path2 = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn2)
        )

        prompt_path3 = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn3)
        )

        prompt_path4 = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn4)
        )

        prompt_path5 = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn5)
        )

        self.prompt1 = parse_text_prompt(prompt_path)
        self.prompt2 = parse_text_prompt(prompt_path2)
        self.prompt3 = parse_text_prompt(prompt_path3)
        self.prompt4 = parse_text_prompt(prompt_path4)
        self.prompt5 = parse_text_prompt(prompt_path5)
        self.plan = None

    def get_action(self, belief, goal: str):
        statistics = {}
        if self.plan is None:
            # No plan yet, we need to come up with one
            ground_plan, statistics = self.full_query_csp(belief, goal)
            if ground_plan is None:
                return None, statistics
            else:
                log.info("Found plan: {}".format(ground_plan))
                self.plan = ground_plan[1:]
                return ground_plan[0], statistics
        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics
        else:
            return None, statistics

    def full_query_csp(self, belief, task):
        _ = self.twin.reset()
        content = "Goal: {}".format(task)
        content = "State: {}\n".format(str(belief)) + content
        chat_history1 = self.prompt1 + [{"role": "user", "content": content}]
        chat_history2 = self.prompt2
        chat_history3 = self.prompt3
        chat_history4 = self.prompt4
        chat_history5 = self.prompt5
        statistics = {}
        statistics["csp_samples"] = 0
        statistics["csp_solve_time"] = 0
        statistics["llm_query_time"] = 0
        for iter in range(self.max_feedbacks + 1):
            statistics["num_feedbacks"] = iter
            st = time.time()
            input_fn = f"llm_high_input_{iter}.txt"
            output_fn = f"llm_high_output_{iter}.txt"
            input_fn2 = f"llm_low_input_{iter}.txt"
            output_fn2 = f"llm_low_output_{iter}.txt"
            input_fn3 = f"llm_symbolic_input_{iter}.txt"
            output_fn3 = f"llm_symbolic_output_{iter}.txt"
            input_fn4 = f"llm_verify_input_{iter}.txt"
            output_fn4 = f"llm_verify_output_{iter}.txt"
            input_fn5 = f"vlm_input_{iter}.txt"
            output_fn5= f"feedback_output_{iter}_vlm.txt"
            
            write_prompt(input_fn, chat_history1)

            # Check if the inputs match
            parent_log_folder = os.path.join(get_log_dir(), "..")
            previous_folder = get_previous_log_folder(parent_log_folder)
            llm_query_time = 0
            if (
                self.use_cache
                and os.path.isfile(os.path.join(previous_folder, output_fn))
                and are_files_identical(
                    os.path.join(previous_folder, input_fn),
                    os.path.join(get_log_dir(), input_fn),
                )
            ):
                log.info("Loading cached LLM response")
                llm_response = read_file(os.path.join(previous_folder, output_fn))
            else:
                log.info("Querying LLM")
                llm_response1, llm_query_time1 = query_llm(chat_history1, seed=self.seed)

                chat_history1.append({"role": "assistant", "content": llm_response1})
                save_log(output_fn, llm_response1)

                subgoals = extract_subgoals_text_only(llm_response1)

                formatted_subgoals = "High-Level Subgoals: " + " ".join(
                    f"{i+1}. {sg.rstrip('.').strip()};" for i, sg in enumerate(subgoals)).rstrip(';') 

                content2 = content + "\n" + formatted_subgoals
                
                chat_history2.append({"role": "user", "content": content2})
                write_prompt(input_fn2, chat_history2)

                llm_response2, llm_query_time2 = query_llm(chat_history2, seed=self.seed)
        

                # chat_history4.append({"role": "user", "content": content})
                # verified = False
                # for i in range(2):
                #     log.info(f"Verification attempt {i+1}")
                #     chat_history4.append({"role": "user", "content": llm_response2})
                #     write_prompt(input_fn4, chat_history4)
                #     verify_response, llm_query_time4 = query_llm(chat_history4, seed=self.seed)

                #     try:
                #         verdict = json.loads(verify_response).get("verdict", "").lower()
                #     except Exception:
                #         verdict = verify_response.strip().lower()

                #     if "pass" in verdict:
                #         verified = True
                #         save_log(output_fn4, verify_response)  # ✅ 修复变量名
                #         break

                #     chat_history2.append({"role": "user", "content": verify_response})
                #     write_prompt(input_fn2, chat_history2)
                #     llm_response2, llm_query_time2 = query_llm(chat_history2, seed=self.seed)
                #     save_log(output_fn2, llm_response2)
                #     save_log(output_fn4, verify_response)
                # if not verified:
                #     continue  # Go to next outer_iter (new high-level subgoal)


                
            statistics["llm_query_time"] += llm_query_time1 + llm_query_time2
            chat_history2.append({"role": "assistant", "content": llm_response2})
            save_log(output_fn2, llm_response2)

            error_message = None
            ground_plan = None

            try:
                llm_code = parse_code(llm_response2)
                exec(llm_code, globals())
                func = globals()[FUNC_NAME]
                domain = globals()[FUNC_DOMAIN]
                st = time.time()
                ground_plan, failure_message, csp_samples = rejection_sample_csp(
                    self.twin,
                    belief,
                    func,
                    domain,
                    max_attempts=self.max_csp_samples,
                )
                statistics["csp_samples"] += csp_samples
                statistics["csp_solve_time"] += time.time() - st

            except Exception as e:
                # Get the traceback as a string
                error_message = traceback.format_exc()
                log.info("Code error: " + str(error_message))

            if ground_plan is not None and error_message is None:
                camera_image, _, _, _, _ = self.twin.get_camera_image_side(image_size=(460 * 2, 640 * 2))
                image_path2 = os.path.join(get_log_dir(), f"twin_frame{iter}.png")
                imageio.imsave(image_path2, camera_image)
                image_path = os.path.join(get_log_dir(), "initial_frame.png")

                with open(image_path, "rb") as image_file:
                    initial_frame = base64.b64encode(image_file.read()).decode("utf-8")
                with open(image_path2, "rb") as image_file:
                    twin_frame = base64.b64encode(image_file.read()).decode("utf-8")

                chat_history5.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Final frame:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{twin_frame}"
                            }
                        },
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                })
                input_fn5 = f"vlm_input_{iter}.txt"
                write_prompt(input_fn5, chat_history5)
                vlm_response, query_time5 = query_llm(chat_history5, seed=self.seed)
                save_log(output_fn5, vlm_response)
                if "yes" in vlm_response.strip().lower():
                    return ground_plan, statistics 
                else:
                    vlm_failure_response = f"VLM check failed: {vlm_response}"
                    content3 = content + "\n" + formatted_subgoals + "\n" + vlm_failure_response + "\n" + llm_response2
                    chat_history5.append({"role": "assistant", "content": content3})
                    chat_history3.append({"role": "user", "content": vlm_failure_response})
                    write_prompt(input_fn3, chat_history3)
                    llm_response3, llm_query_time3 = query_llm(chat_history3, seed=self.seed)
                    chat_history1.append({"role": "user", "content": llm_response3})
                    save_log(output_fn3, llm_response3)
                #return ground_plan, statistics 
            else:

                if error_message is not None:
                    failure_response = error_message
                else:
                    failure_response = ""
                    for fm, count in failure_message.most_common(2):
                        failure_response += f"{count} occurences: {fm}\n"

                save_log(f"feedback_output_{iter}.txt", failure_response)

                content3 = content + "\n" + formatted_subgoals + "\n" + failure_response + "\n" + llm_response2
                chat_history3.append({"role": "user", "content": content3})
                #chat_history3.append({"role": "user", "content": failure_response})
                write_prompt(input_fn3, chat_history3)
                llm_response3, llm_query_time3 = query_llm(chat_history3, seed=self.seed)
                chat_history3.append({"role": "assistant", "content": llm_response3})
                chat_history1.append({"role": "user", "content": llm_response3})
                save_log(output_fn3, llm_response3)
                # match = re.search(r"\*\*\*rule-based symbolic constraint\*\*\*:\s*(.*)", llm_response3, re.DOTALL)
                # if match:
                #     constraint_text = match.group(1).strip()
                # else:
                #     constraint_text = None

        return ground_plan, statistics


