# Rebuild the illustrative figure in docs/gain-fraction.md.
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "docs/assets/gain-fraction.png"

history <- data.frame(
  queries = c(100, 400, 800, 1200, 1600, 2300, 3000, 3600, 4500, 5400),
  mean_score = c(0.20, 0.40, 0.63, 0.71, 0.84, 0.93, 0.98, 1, 1, 1)
)
initial <- history$mean_score[1]
plateau <- tail(history$mean_score, 1)
targets <- initial + c(0.5, 0.9) * (plateau - initial)
first_hit <- vapply(targets, function(value) which(history$mean_score >= value)[1], integer(1))
selected <- transform(history[first_hit, ], target = targets, label = c("50% set", "90% set"))
labels <- data.frame(
  y = c(plateau, rev(targets), initial),
  label = c("Plateau  1.00", "90% target  0.92", "50% target  0.60", "Initial mean  0.20")
)

accent <- "#007D78"
trace_color <- "#234666"
figure <- ggplot(history, aes(queries, mean_score)) +
  annotate("segment", x = 0, xend = 5500, y = c(initial, plateau),
           yend = c(initial, plateau), linewidth = 0.35, colour = "#A7ACB2") +
  geom_segment(data = selected, aes(x = 0, xend = 5500, y = target, yend = target),
               inherit.aes = FALSE, linewidth = 0.35, linetype = "dashed", colour = accent) +
  geom_step(linewidth = 0.8, colour = trace_color) +
  geom_segment(data = selected, aes(xend = queries, y = 0.27, yend = mean_score),
               linewidth = 0.35, linetype = "dashed", colour = accent) +
  geom_point(data = selected, size = 2.6, colour = accent) +
  geom_text(data = selected, aes(x = queries + 115, y = 0.28, label = label),
            inherit.aes = FALSE, colour = accent, hjust = 0, size = 3, family = "sans") +
  geom_text(data = labels, aes(x = 5630, y = y, label = label),
            inherit.aes = FALSE, colour = "#343A43", hjust = 0, size = 3, family = "sans") +
  scale_x_continuous(breaks = seq(0, 5000, 1000), limits = c(0, 7100), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0.2, 1, 0.2), limits = c(0.16, 1.07), expand = c(0, 0)) +
  labs(x = "Optimizer queries", y = "Mean objective score of retained set") +
  theme_classic(base_size = 10, base_family = "sans") +
  theme(axis.line = element_line(linewidth = 0.35, colour = "#333333"),
        axis.ticks = element_line(linewidth = 0.35, colour = "#333333"),
        plot.margin = margin(7, 10, 7, 7))

# The optional PDF review copy uses the same dimensions as the documentation image.
ggsave(output, figure, width = 165, height = 86, units = "mm", dpi = 300, bg = "white")
if (length(args) > 1) {
  ggsave(paste0(args[[2]], ".pdf"), figure, device = grDevices::pdf, width = 165,
         height = 86, units = "mm", bg = "white", useDingbats = FALSE)
}
