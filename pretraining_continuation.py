import hashlib
import json
import math
from pathlib import Path


def build_schedule(mode, total_steps, start_lr, peak_lr, warmup_steps=0,
                   decay_steps=0):
    counts = (total_steps, warmup_steps, decay_steps)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Schedule lengths must be nonnegative integers")
    if total_steps == 0:
        raise ValueError("Continuation requires at least one update")
    if any(not math.isfinite(value) or value <= 0
           for value in (start_lr, peak_lr)):
        raise ValueError("Learning rates must be positive and finite")
    if mode == "constant":
        if peak_lr != start_lr or warmup_steps or decay_steps:
            raise ValueError("Constant continuation cannot warm up or decay")
        return [start_lr] * total_steps
    if mode != "rewarm":
        raise ValueError("Unknown continuation schedule: %s" % mode)
    if peak_lr <= start_lr or warmup_steps < 2 or decay_steps < 2:
        raise ValueError("Rewarming requires a higher peak and nonempty ramps")
    stable_steps = total_steps - warmup_steps - decay_steps
    if stable_steps < 1:
        raise ValueError("Rewarming requires a positive stable interval")
    warmup = [start_lr + (peak_lr - start_lr) * index / (warmup_steps - 1)
              for index in range(warmup_steps)]
    decay = [start_lr + 0.5 * (peak_lr - start_lr) *
             (1.0 + math.cos(math.pi * index / (decay_steps - 1)))
             for index in range(decay_steps)]
    return warmup + [peak_lr] * stable_steps + decay


def phase_step(epoch, step, steps_per_epoch, start_epoch, phase_epochs):
    values = (epoch, step, steps_per_epoch, start_epoch, phase_epochs)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Training counters must be nonnegative integers")
    if steps_per_epoch == 0 or phase_epochs == 0 or step > steps_per_epoch:
        raise ValueError("Invalid continuation budget or step within epoch")
    offset = (epoch - start_epoch) * steps_per_epoch + step
    if epoch < start_epoch or offset > phase_epochs * steps_per_epoch:
        raise ValueError("Checkpoint lies outside the continuation phase")
    return offset


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_arguments(arguments, expected):
    actual = arguments if isinstance(arguments, dict) else vars(arguments)
    mismatches = [name for name, value in expected.items()
                  if name not in actual or actual[name] != value]
    if mismatches:
        raise ValueError("Continuation training contract differs: %s" %
                         ", ".join(sorted(mismatches)))


def _resume_candidates(output_dir):
    rolling = [output_dir / name for name in
               ("checkpoint.pth", "checkpoint_alt.pth")]
    rolling = sorted((path for path in rolling if path.is_file()),
                     key=lambda path: path.stat().st_mtime_ns, reverse=True)
    numbered = [path for path in output_dir.glob("checkpoint-*.pth")
                if path.stem.removeprefix("checkpoint-").isdigit()]
    numbered.sort(key=lambda path: int(path.stem.split("-")[-1]), reverse=True)
    return rolling + numbered


def restore_continuation(args, model, optimizer, loss_scaler, steps_per_epoch,
                         world_size, checkpoint_loader=None):
    plan = json.loads(Path(args.continuation_plan).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or args.continuation_arm not in plan["arms"]:
        raise ValueError("Unknown continuation plan or arm")
    if args.resume or not args.auto_resume or args.no_mid_epoch_resume:
        raise ValueError("Continuation owns resume selection and requires step resume")
    if args.gradient_accumulation_steps != 1:
        raise ValueError("Continuation currently requires one batch per update")
    if world_size != plan["world_size"] or steps_per_epoch != plan["steps_per_epoch"]:
        raise ValueError("World size or epoch length differs from the frozen plan")
    if args.batch_size * world_size != plan["global_batch_size"]:
        raise ValueError("Continuation must preserve the original global batch")
    _check_arguments(args, plan["training_args"])
    start_epoch = plan["source_epoch"] + 1
    phase_epochs = plan["phase_epochs"]
    if args.epochs != start_epoch + phase_epochs:
        raise ValueError("Total epochs must equal parent epochs plus phase budget")
    phase_step(start_epoch, 0, steps_per_epoch, start_epoch, phase_epochs)
    source_path = Path(plan["source_checkpoint"])
    output_dir = Path(args.output_dir)
    if output_dir != Path(plan["outputs"][args.continuation_arm]):
        raise ValueError("Continuation output does not match its arm")
    if output_dir == source_path.parent or source_path.parent in output_dir.parents:
        raise ValueError("Continuation cannot write into its parent output")
    contract = {"plan": plan, "arm": args.continuation_arm}
    schedule = build_schedule(
        total_steps=phase_epochs * steps_per_epoch,
        **plan["arms"][args.continuation_arm],
    )
    if not math.isfinite(plan["weight_decay"]) or plan["weight_decay"] < 0:
        raise ValueError("Invalid continuation weight decay")
    if checkpoint_loader is None:
        import torch
        checkpoint_loader = lambda path: torch.load(
            path, map_location="cpu", weights_only=False)

    candidates = _resume_candidates(output_dir)
    is_parent = not candidates
    if is_parent:
        if (output_dir / "log.txt").is_file():
            raise FileNotFoundError("Continuation logs exist but no checkpoint survives")
        if file_sha256(source_path) != plan["source_sha256"]:
            raise ValueError("Parent checkpoint SHA256 differs from the frozen plan")
        candidates = [source_path]
    checkpoint = None
    for candidate in candidates:
        try:
            checkpoint = checkpoint_loader(candidate)
        except Exception as error:
            print("Unreadable continuation checkpoint %s: %s" % (candidate, error),
                  flush=True)
            continue
        if not isinstance(checkpoint, dict) or not all(
                key in checkpoint for key in ("model", "optimizer", "epoch", "args")):
            raise ValueError("Continuation requires a full training checkpoint")
        if not checkpoint["optimizer"].get("state"):
            raise ValueError("Continuation checkpoint has no trained optimizer state")
        saved_args = checkpoint["args"]
        saved_args = saved_args if isinstance(saved_args, dict) else vars(saved_args)
        _check_arguments(saved_args, plan["training_args"])
        saved_contract = saved_args.get("continuation_contract")
        epoch = checkpoint["epoch"]
        step = checkpoint.get("step_in_epoch", 0)
        if type(epoch) is not int or type(step) is not int:
            raise ValueError("Checkpoint progress must contain integer counters")
        if is_parent:
            if saved_contract is not None or epoch != plan["source_epoch"] or step != 0:
                raise ValueError("Parent must be the specified completed epoch")
            next_epoch, next_step = start_epoch, 0
        else:
            if saved_contract != contract:
                raise ValueError("Checkpoint belongs to another continuation contract")
            phase_step(epoch, step, steps_per_epoch, start_epoch, phase_epochs)
            if epoch >= start_epoch + phase_epochs:
                raise ValueError("Saved epoch exceeds the continuation budget")
            if step == 0 or step == steps_per_epoch:
                next_epoch, next_step = epoch + 1, 0
            else:
                next_epoch, next_step = epoch, step
        offset = phase_step(next_epoch, next_step, steps_per_epoch,
                            start_epoch, phase_epochs)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler is not None and "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        args.resume = str(candidate)
        args.start_epoch = next_epoch
        args.start_step_in_epoch = next_step
        args.continuation_contract = contract
        print("CONTINUATION_STATE " + json.dumps({
            "arm": args.continuation_arm,
            "source": str(candidate),
            "parent_sha256": plan["source_sha256"],
            "loaded_parent": is_parent,
            "optimizer_restored": True,
            "start_epoch": next_epoch,
            "start_step_in_epoch": next_step,
            "phase_step": offset,
            "phase_total_steps": len(schedule),
            "next_lr": schedule[offset] if offset < len(schedule) else None,
        }, sort_keys=True), flush=True)
        return schedule, [plan["weight_decay"]] * len(schedule), start_epoch
    raise RuntimeError("No readable continuation checkpoint; refusing scratch or parent rollback")